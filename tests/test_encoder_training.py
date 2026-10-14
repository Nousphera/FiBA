import importlib
import torch
from fiba.encoder_poisoning import EncoderAttack


def test_encoder_trains_and_checkpoint_loads(tmp_path):
    training = importlib.import_module('fiba.encoder_training')
    checkpoint = tmp_path/'encoder.pt'
    report = training.main(['--synthetic','--epochs','1','--batch-size','2','--output',str(checkpoint),'--device','cpu'])
    assert checkpoint.is_file()
    assert report['epochs'] == 1
    assert report['synthetic'] is True
    attack = EncoderAttack([0],[(0,0,32,32)],0,1,False,None,False,False,checkpoint=checkpoint)
    poisoned = attack.embed_trigger_in_image([torch.zeros(3,32,32)],[0],attack)
    assert poisoned[0][0].shape == (3,32,32)
    assert torch.isfinite(poisoned[0][0]).all()
    assert poisoned[0][1] == 1
