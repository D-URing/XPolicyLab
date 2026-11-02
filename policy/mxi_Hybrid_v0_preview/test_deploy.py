"""Contract tests with doubles; not robot-success evidence."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('mxi_deploy',Path(__file__).with_name('deploy.py'))
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class Env:
    step_lim = 26
    def __init__(self,end=26,fail_at=None): self.n=0; self.end=end; self.fail_at=fail_at
    def is_episode_end(self): return self.n >= self.end
    def get_obs(self): return {'frame':self.n}
    def take_action(self, action):
        if self.n==self.fail_at: raise RuntimeError('execution failure')
        self.n+=1


class Client:
    def __init__(self): self.events=[]; self.n=0
    def call(self,func_name,obs):
        self.events.append((func_name,obs))
        if func_name=='propose_actions':
            return {'actions':[{}]*min(24,26-self.n),'receipt':{'episode_id':'e','chunk_id':self.n}}
        if func_name=='acknowledge':
            self.n+=obs['executed_steps']
            return {'total_executed':self.n,'ended':obs['reason']=='episode_end'}
        return {}


class DeploymentTests(unittest.TestCase):
    def test_every_step_observed_once_and_budget_truncates(self):
        env,client=Env(),Client()
        deploy.eval_one_episode(env,client)
        self.assertEqual([o['frame'] for n,o in client.events if n=='update_obs'],list(range(26)))
        self.assertEqual([o['executed_steps'] for n,o in client.events if n=='acknowledge'],[24,2])
        self.assertEqual(client.events[-1][0],'close_client')

    def test_terminal_receipt(self):
        env,client=Env(end=3),Client()
        deploy.eval_one_episode(env,client)
        receipts=[o for n,o in client.events if n=='acknowledge']
        self.assertEqual((receipts[0]['executed_steps'],receipts[0]['reason']),(3,'episode_end'))

    def test_execution_error_not_retried(self):
        env,client=Env(fail_at=2),Client()
        with self.assertRaisesRegex(RuntimeError,'execution failure'): deploy.eval_one_episode(env,client)
        receipts=[o for n,o in client.events if n=='acknowledge']
        self.assertEqual((receipts[0]['executed_steps'],receipts[0]['reason']),(2,'interrupted'))
        self.assertEqual(sum(n=='propose_actions' for n,o in client.events),1)

    def test_missing_budget_refused_before_rpc(self):
        env,client=Env(),Client();env.step_lim=None
        with self.assertRaises(ValueError):deploy.eval_one_episode(env,client)
        self.assertFalse(client.events)


if __name__=='__main__':unittest.main()
