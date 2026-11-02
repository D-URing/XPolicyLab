"""Small XPolicyLab boundary; implementation and assets are supplied separately."""
import json
import os
from pathlib import Path
from uuid import uuid4

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        if model_cfg.get('env_cfg_type') != 'arx_x5' or model_cfg.get('action_type') != 'ee':
            raise ValueError('MXI s4000 requires arx_x5 / absolute EE actions')
        if model_cfg.get('eval_batch', False):
            raise ValueError('Vectorized episodes are unsupported; use independent servers')
        info = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        if len(info['arm_dim']) != 2 or info['ee_dim'] != [1, 1]:
            raise ValueError('Expected two arms with one gripper coordinate each')
        root = resolve_checkpoint_root(model_cfg, Path(__file__).parent/'checkpoints')
        config = json.loads((root/'runtime.json').read_text())
        if config.get('profile') != 'mxi_Hybrid_v0_preview' or config.get('checkpoint_step') != 4000:
            raise ValueError('Wrong deployment profile or checkpoint')
        for key in ('joint_checkpoint', 'qwen_base'):
            config[key] = str((root/config[key]).resolve())
        for key in ('ckpt_name', 'checkpoint_path', 'video_checkpoint_path',
                    'dataset_stats_path', 'wan_model_dir'):
            config['native'][key] = str((root/config['native'][key]).resolve())
        expected = dict(replan_steps=24, num_inference_steps=10, seed=1,
                        mixed_precision='bf16', sigma_shift=1.0, action_horizon=32,
                        env_cfg_type='arx_x5', action_type='ee')
        if config.get('decode_mode') != 'schema' or any(config['native'].get(k) != v for k,v in expected.items()):
            raise ValueError('Bundle inference contract differs from the frozen s4000 profile')
        output = Path(os.environ.get('MXI_OUTPUT_ROOT', './mxi-results')).resolve()
        config.update(output_root=str(output), trace_path=str(output/f'decisions-{uuid4().hex}.jsonl'),
                      max_clients=1)
        from mxi.adapters.robodojo_online import Model as Core
        self.core = Core(config)
        self._tag = None

    def set_request_context(self, context):
        self.core.set_request_context(context)

    def reset_client(self, obs):
        result = self.core.reset_client(obs)
        self._tag = dict(obs)
        return result

    def close_client(self, obs):
        result = self.core.close_client(obs)
        self._tag = None
        return result

    def update_obs(self, obs):
        return self.core.update_obs(obs)

    def propose_actions(self, obs):
        """Actions AND receipt; the custom deployment loop acknowledges execution."""
        return self.core.get_action(obs)

    def get_action(self):
        if self._tag is None:
            raise RuntimeError('reset_client with the official budget is required')
        return self.core.get_action(self._tag)['actions']

    def acknowledge(self, obs):
        return self.core.acknowledge(obs)

    def reset(self):
        return self.core.reset()

    def update_obs_batch(self, obs_list):
        raise NotImplementedError('Use one episode per server')

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError('Use one episode per server')
