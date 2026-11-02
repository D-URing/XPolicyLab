"""Real official WebSocket transport with a test double, NOT a model/robot test."""
import asyncio
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from client_server.ws import WsModelClient
from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from XPolicyLab.utils.process_data import encode_image_bit

spec=importlib.util.spec_from_file_location('mxi_deploy',Path(__file__).with_name('deploy.py'))
deploy=importlib.util.module_from_spec(spec);spec.loader.exec_module(deploy)


class TransportDouble:
    def __init__(self):self.frames=[];self.executed=0;self.closed=False
    def reset_client(self,obs):self.executed=0;return {'reset':True}
    def update_obs(self,obs):
        for camera in obs['vision'].values():
            image=camera['color']
            assert isinstance(image,np.ndarray) and image.dtype==np.uint8
            assert image[...,0].mean()>240 and image[...,2].mean()<10
        self.frames.append(obs['frame']);return {}
    def propose_actions(self,obs):
        return {'actions':[{'test_only':np.ones(1,dtype=np.float32)}]*3,
                'receipt':{'episode_id':'test','chunk_id':1}}
    def acknowledge(self,obs):
        self.executed+=obs['executed_steps']
        return {'total_executed':self.executed,'ended':obs['reason']=='episode_end'}
    def close_client(self,obs):self.closed=True;return {'closed':True}


class TransportEnvironment:
    step_lim=3
    def __init__(self):self.n=0
    def get_obs(self):
        rgb=np.zeros((16,16,3),dtype=np.uint8);rgb[...,0]=255
        encoded=encode_image_bit(rgb)
        return {'frame':self.n,'vision':{
            'cam_head':{'color':rgb},'cam_left_wrist':{'color':np.array(bytearray(encoded),dtype=np.uint8)},
            'cam_right_wrist':{'color':encoded}}}
    def is_episode_end(self):return self.n==3
    def take_action(self,action):
        assert isinstance(action['test_only'],np.ndarray)
        self.n+=1


class WebsocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_decoding_and_receipts_over_official_server(self):
        model=TransportDouble()
        server=PolicyServer(model,PolicyServerConfig(host='127.0.0.1',port=0))
        await server.start()
        def run():
            client=WsModelClient(url=server.url,evaluation_id='unit',trial_id='unit',
                                 request_timeout_s=10,max_connect_attempts=1)
            try:deploy.eval_one_episode(TransportEnvironment(),client)
            finally:client.close()
        try:await asyncio.to_thread(run)
        finally:await server.stop()
        self.assertEqual(model.frames,[0,1,2])
        self.assertEqual(model.executed,3)
        self.assertTrue(model.closed)


if __name__=='__main__':unittest.main()
