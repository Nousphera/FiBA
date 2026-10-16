"""Native Flower Message identity and reply validation."""
import pytest
from flwr.app import ConfigRecord, Message, MetricRecord, RecordDict


@pytest.fixture(autouse=True)
def flower_task_identity(monkeypatch):
    from flwr.supercore.task_identity import TaskIdentity
    monkeypatch.setattr(TaskIdentity,'_run_id',1)
    monkeypatch.setattr(TaskIdentity,'_task_id',1)
    monkeypatch.setattr(TaskIdentity,'_node_id',1)


def instruction(node, rnd=1):
    return Message(RecordDict({'config':ConfigRecord({'round':rnd,'phase':'train'})}),
                   dst_node_id=node,message_type='train',group_id=str(rnd))


def test_reply_validation_rejects_missing_and_duplicate_replies():
    from fiba.transport import replies
    message = instruction(19)
    reply = Message(RecordDict({'metrics':MetricRecord({'client_id':0})}),reply_to=message)
    class Grid:
        def send_and_receive(self,*args,**kwargs):
            return iter([reply,reply])
    with pytest.raises(RuntimeError,match='one successful'):
        list(replies(Grid(),[message],1))
    class EmptyGrid:
        def send_and_receive(self,*args,**kwargs):
            return iter([])
    with pytest.raises(RuntimeError,match='Missing'):
        list(replies(EmptyGrid(),[message],1))


def test_handshake_maps_logical_partitions_independently_of_node_order():
    from fiba.transport import identify
    class Grid:
        def send_and_receive(self,messages,**kwargs):
            mapping = {51:1,13:0}
            return [Message(RecordDict({'metrics':MetricRecord({'client_id':mapping[m.metadata.dst_node_id]})}),reply_to=m)
                    for m in reversed(messages)]
    assert identify(Grid(),[51,13],2,1) == {0:13,1:51}


def test_handshake_rejects_duplicate_partitions():
    from fiba.transport import identify
    class Grid:
        def send_and_receive(self,messages,**kwargs):
            return [Message(RecordDict({'metrics':MetricRecord({'client_id':0})}),reply_to=m) for m in messages]
    with pytest.raises(ValueError,match='unique'):
        identify(Grid(),[51,13],2,1)


def test_native_messages_match_sequential_training_and_checkpoint(tmp_path,monkeypatch):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from fiba import data
    from fiba.client_app import train
    from fiba.config import defaults
    from fiba.server_app import run_experiment
    from torch.utils.data import Dataset
    class Images(Dataset):
        targets = np.arange(8)
        classes = list(range(10))
        def __len__(self):
            return 8
        def __getitem__(self,index):
            return torch.full((3,32,32),.4+index*.01),index
    monkeypatch.setattr(data,'load_datasets',lambda *args:(Images(),Images()))
    cfg = defaults() | dict(num_clients=2,num_rounds=1,local_epochs=1,batch_size=3,
                            split='iid',device='cpu',attack='none',attacker_fraction=0.,
                            output_dir=str(tmp_path/'sequential'))
    sequential = run_experiment(cfg)
    native_cfg = cfg | {'output_dir':str(tmp_path/'native')}
    class Grid:
        def get_node_ids(self):
            return [91,13]
        def send_and_receive(self,messages,**kwargs):
            result = []
            for message in reversed(messages):
                cid = {91:0,13:1}[message.metadata.dst_node_id]
                context = SimpleNamespace(node_config={'partition-id':cid},run_config=native_cfg)
                result.append(train(message,context))
            return result
    native = run_experiment(native_cfg,Grid())
    assert native['rounds'][0]['accuracy'] == sequential['rounds'][0]['accuracy']
    assert native['rounds'][0]['clients'] == sequential['rounds'][0]['clients']
    left = torch.load(tmp_path/'sequential/checkpoint.pt',weights_only=True)['model']
    right = torch.load(tmp_path/'native/checkpoint.pt',weights_only=True)['model']
    assert all(torch.equal(value,right[key]) for key,value in left.items())
