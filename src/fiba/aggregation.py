"""Finite, schema-checked federated aggregation on model state dictionaries.

Defenses operate on floating model coordinates. Integral state (for example
BatchNorm batch counters) takes the coordinate maximum of the contributing
states; it is never included in similarity/distance calculations or scaling.
"""
import math
import torch


def attacker_ids(cfg):
    """Stable attacker identities shared by training and trigger partitioning."""
    n = int(cfg.get('num_clients', 10))
    fraction = float(cfg.get('attacker_fraction', .1))
    if n < 1 or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('num_clients must be positive and attacker_fraction must be in [0,1]')
    return list(range(math.floor(n * fraction)))


class Aggregator:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.method = cfg.get('aggregation', 'fedavg')
        if self.method not in {'fedavg', 'median', 'trimmed-mean', 'foolsgold', 'krum', 'multi-krum'}:
            raise ValueError(f'Unknown aggregation: {self.method}')
        self.history = {}
        self.last_weights = {}

    def aggregate(self, states, counts, reference, client_ids, *, round_number=1):
        """Aggregate complete client states relative to their common reference.

        Model replacement simulates informed attackers before the configured
        defense: on selected rounds, their delta is multiplied by the inverse
        participating sample weight. BatchNorm running means/variances remain
        unscaled to preserve their buffer semantics and valid variances.
        With one attacker and FedAvg this gives
        the canonical replacement update; it is an explicit oracle assumption.
        """
        n = len(states)
        if n < 1 or len(counts) != n or len(client_ids) != n or len(set(client_ids)) != n:
            raise ValueError('Aggregation requires nonempty equally sized states/counts/unique client IDs')
        if not reference:
            raise ValueError('Reference state cannot be empty')
        if any(not isinstance(value, torch.Tensor) for value in reference.values()):
            raise ValueError('Reference model state must contain tensors')
        if any(isinstance(c, bool) or int(c) != c or c <= 0 for c in counts):
            raise ValueError('Client example counts must be positive integers')
        for state in [reference, *states]:
            if set(state) != set(reference):
                raise ValueError('Client model schema differs from reference')
            for key, value in state.items():
                base = reference[key]
                if not isinstance(value, torch.Tensor) or value.shape != base.shape or value.dtype != base.dtype:
                    raise ValueError(f'Client tensor schema differs for {key}')
                if value.is_complex() or (value.is_floating_point() and not torch.isfinite(value).all()):
                    raise ValueError(f'Client contains non-finite or complex tensor: {key}')
                if key.endswith('running_var') and torch.any(value < 0):
                    raise ValueError(f'Invalid negative BatchNorm running variance: {key}')
        floating = [key for key, tensor in reference.items() if tensor.is_floating_point()]
        if not floating:
            raise ValueError('Aggregation requires floating model parameters')
        # CPU float64 avoids overflow and device/dtype variation in robust scores.
        working = [{key: value.detach().cpu().to(torch.float64) if value.is_floating_point()
                    else value.detach().cpu().clone() for key, value in state.items()} for state in states]
        base = {key: value.detach().cpu().to(torch.float64) for key, value in reference.items() if key in floating}
        replacement = self.cfg.get('model_replacement', False)
        rounds = self.cfg.get('replacement_rounds', '30,60,90')
        rounds = [int(r.strip()) for r in rounds.split(',') if r.strip()] if isinstance(rounds, str) else list(rounds)
        if replacement and round_number in rounds:
            malicious = set(attacker_ids(self.cfg))
            for state, count, cid in zip(working, counts, client_ids):
                if cid in malicious:
                    factor = sum(counts) / count
                    for key in floating:
                        if key.endswith(('running_mean', 'running_var')):
                            continue
                        state[key] = base[key] + factor * (state[key] - base[key])
                        if not torch.isfinite(state[key]).all():
                            raise ValueError('Model replacement produced non-finite coordinates')
        indices = list(range(n))
        weights = torch.tensor(counts, dtype=torch.float64)
        weights /= weights.sum()
        if self.method in {'krum', 'multi-krum'}:
            f = int(self.cfg.get('byzantine_clients', len(attacker_ids(self.cfg))))
            if f < 0 or n < 2 * f + 3:
                raise ValueError('Krum requires n >= 2*f+3 participating clients')
            vectors = torch.stack([torch.cat([state[key].reshape(-1) for key in floating]) for state in working])
            distances = torch.cdist(vectors, vectors).square()
            distances.fill_diagonal_(float('inf'))
            neighbors = n - f - 2
            scores = distances.sort(dim=1, stable=True).values[:, :neighbors].sum(1)
            keep = 1 if self.method == 'krum' else int(self.cfg.get('krum_keep', 1))
            if not 1 <= keep <= neighbors:
                raise ValueError('Multi-Krum krum_keep must be between 1 and n-f-2')
            indices = torch.argsort(scores, stable=True)[:keep].tolist()
            # The original Multi-Krum rule averages selected vectors uniformly.
            weights = torch.ones(keep, dtype=torch.float64) / keep
        elif self.method == 'foolsgold':
            classifier = [key for key in floating if key.endswith(('fc.weight', 'classifier.weight', 'linear.weight'))]
            if len(classifier) != 1:
                raise ValueError('FoolsGold requires exactly one fc.weight, classifier.weight or linear.weight tensor')
            key = classifier[0]
            for state, cid in zip(working, client_ids):
                delta = (state[key] - base[key]).reshape(-1)
                self.history[cid] = self.history.get(cid, torch.zeros_like(delta)) + delta
            # Compare against all historical clients, including nonparticipants.
            history_ids = sorted(self.history)
            vectors = torch.stack([self.history[cid] for cid in history_ids])
            trust = self._foolsgold(vectors)
            lookup = {cid: trust[i] for i, cid in enumerate(history_ids)}
            weights = torch.stack([lookup[cid] for cid in client_ids])
            if weights.sum() <= 0:
                raise ValueError('FoolsGold assigned no positive trust to participating clients')
            weights /= weights.sum()
        self.last_weights = ({} if self.method in {'median', 'trimmed-mean'} else
                             {int(client_ids[i]): float(weight) for i, weight in zip(indices, weights)})
        result = {}
        for key, tensor in reference.items():
            values = torch.stack([working[index][key] for index in indices])
            if not tensor.is_floating_point():
                output = values.amax(0)
            elif self.method == 'median':
                ordered = values.sort(dim=0).values
                output = (ordered[(n - 1) // 2] + ordered[n // 2]) / 2
            elif self.method == 'trimmed-mean':
                fraction = float(self.cfg.get('trim_fraction', .1))
                if not math.isfinite(fraction) or not 0 <= fraction < .5:
                    raise ValueError('trim_fraction must be in [0,.5)')
                trim = int(n * fraction)
                output = values.sort(dim=0).values[trim:n-trim].mean(0)
            else:
                shape = (-1,) + (1,) * tensor.ndim
                output = (values * weights.reshape(shape)).sum(0)
            output = output.to(device=tensor.device, dtype=tensor.dtype)
            if output.is_floating_point() and not torch.isfinite(output).all():
                raise ValueError(f'Aggregated tensor overflowed: {key}')
            result[key] = output
        return result

    @staticmethod
    def _foolsgold(vectors):
        n = len(vectors)
        if n == 1:
            return torch.ones(1, dtype=torch.float64)
        unit = torch.nn.functional.normalize(vectors, dim=1)
        similarity = (unit @ unit.T).clamp(-1, 1)
        similarity.fill_diagonal_(0)
        maximum = similarity.amax(1)
        for i in range(n):
            for j in range(n):
                if i != j and maximum[i] < maximum[j] and maximum[j] > 0:
                    similarity[i, j] *= maximum[i] / maximum[j]
        trust = (1 - similarity.amax(1)).clamp(0, 1)
        if trust.max() <= 0:
            return torch.zeros_like(trust)
        trust /= trust.max()
        trust = trust.clamp(max=.99)
        # Zero trust is intentionally preserved without log(0) warnings.
        positive = trust > 0
        result = torch.zeros_like(trust)
        result[positive] = (torch.log(trust[positive] / (1-trust[positive])) + .5).clamp(0, 1)
        return result
