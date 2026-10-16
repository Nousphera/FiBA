"""Validated native Flower Message routing by stable logical partition identity."""
import time
from flwr.app import ConfigRecord, Message, RecordDict


def wait_for_nodes(grid, count, timeout):
    deadline = time.monotonic() + timeout
    while True:
        nodes = sorted(grid.get_node_ids())
        if len(set(nodes)) != len(nodes) or len(nodes) > count:
            raise ValueError('Unexpected Flower SuperNodes')
        if len(nodes) == count:
            return nodes
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Only {len(nodes)}/{count} Flower SuperNodes registered')
        time.sleep(.25)


def replies(grid, messages, timeout):
    expected = {message.metadata.dst_node_id:message for message in messages}
    if len(expected) != len(messages):
        raise ValueError('Duplicate destination nodes')
    seen = set()
    for reply in grid.send_and_receive(messages,timeout=timeout):
        node = reply.metadata.src_node_id
        if reply.has_error() or node not in expected or node in seen:
            raise RuntimeError('Each selected client must return one successful update')
        if reply.metadata.group_id != expected[node].metadata.group_id:
            raise ValueError('Reply round mismatch')
        seen.add(node)
        yield reply
    if seen != set(expected):
        raise RuntimeError('Missing client replies; round was not aggregated')


def identify(grid, nodes, count, timeout):
    messages = [Message(RecordDict({'config':ConfigRecord({'round':0,'phase':'identify'})}),
                        dst_node_id=node,message_type='train',group_id='0') for node in nodes]
    mapping = {}
    for reply in replies(grid,messages,timeout):
        cid = reply.content['metrics']['client_id']
        if type(cid) != int or cid in mapping or not 0 <= cid < count:
            raise ValueError('Client partition IDs must be unique and complete')
        mapping[cid] = reply.metadata.src_node_id
    if sorted(mapping) != list(range(count)):
        raise ValueError('Client partition IDs must exactly cover configured clients')
    return mapping
