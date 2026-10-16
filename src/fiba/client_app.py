"""Native Flower Message client; the same trainer powers sequential execution."""
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from .training import train_client

app = ClientApp()


@app.train()
def train(message, context):
    from .config import validate
    cfg = validate(dict(context.run_config))
    cid = context.node_config['partition-id']
    if type(cid) != int or not 0 <= cid < cfg['num_clients']:
        raise ValueError('Invalid client partition')
    incoming = message.content['config']
    rnd, phase = incoming['round'], incoming['phase']
    if type(rnd) != int or phase not in ('identify','train') or (phase == 'identify') != (rnd == 0):
        raise ValueError('Invalid experiment message')
    if phase == 'identify':
        return Message(RecordDict({'metrics':MetricRecord({'client_id':cid})}),reply_to=message)
    mask = None
    if 'attention' in message.content:
        mask = message.content['attention'].to_torch_state_dict()['mask']
    state, metrics = train_client(message.content['model'].to_torch_state_dict(),cfg,cid,rnd,mask)
    mask = metrics.pop('_attention_mask',None)
    content = RecordDict({'model':ArrayRecord(state),'metrics':MetricRecord(metrics)})
    if mask is not None:
        content['attention'] = ArrayRecord({'mask':mask})
    return Message(content,reply_to=message)
