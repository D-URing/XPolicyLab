"""Receipt-aware loop; no model dependencies are needed on the simulator machine."""
import logging
import os
import sys
from uuid import uuid4


def policy_observation(env, tag):
    obs = env.get_obs()
    result = {key: obs[key] for key in ('state','instruction','instructions',
              'task_instruction','env_idx','data_format_version') if key in obs}
    if 'state' not in result or not any(result.get(k) for k in
            ('instruction','instructions','task_instruction')):
        raise ValueError('Raw robot state and original task instruction are required')
    # Do not forward simulator annotations, depth, object poses or task metrics.
    result['vision'] = {name: {'color':obs['vision'][name]['color']} for name in
                       ('cam_head','cam_left_wrist','cam_right_wrist')}
    return {**result, **tag}


def eval_one_episode(TASK_ENV, model_client):
    budget = getattr(TASK_ENV, 'step_lim', None)
    # Only the official debug client uses this attribute; never substitute it
    # for a missing simulator budget.
    if budget is None and os.environ.get('EVAL_ENV_TYPE') == 'debug':
        if type(TASK_ENV).__name__ == 'TestEnv':
            budget = getattr(TASK_ENV, 'episode_step_limit', None)
    if type(budget) is not int or budget <= 0:
        raise ValueError('Official positive integer step_lim is required')
    tag = {'_mxi_native_client_id': uuid4().hex, '_step_budget': budget}
    model_client.call(func_name='reset_client', obs=tag)
    try:
        total = 0
        while not TASK_ENV.is_episode_end():
            model_client.call(func_name='update_obs', obs=policy_observation(TASK_ENV,tag))
            packet = model_client.call(func_name='propose_actions', obs=tag)
            actions, receipt = packet['actions'], packet['receipt']
            if not 0 < len(actions) <= min(24, budget-total):
                raise ValueError('Invalid action chunk length; refusing execution')
            executed, reason = 0, 'interrupted'
            try:
                for i, action in enumerate(actions):
                    if TASK_ENV.is_episode_end():
                        break
                    TASK_ENV.take_action(action)
                    executed += 1
                    if TASK_ENV.is_episode_end():
                        break
                    if i+1 < len(actions):
                        model_client.call(func_name='update_obs', obs=policy_observation(TASK_ENV,tag))
                reason = 'episode_end' if TASK_ENV.is_episode_end() else 'consumed'
            finally:
                result = model_client.call(func_name='acknowledge', obs={**tag,
                    'episode_id': receipt['episode_id'], 'chunk_id': receipt['chunk_id'],
                    'executed_steps': executed, 'reason': reason})
            total += executed
            if result['total_executed'] != total:
                raise RuntimeError('Execution accounting mismatch')
            if result['ended']:
                if not TASK_ENV.is_episode_end():
                    raise RuntimeError('Policy ended before the official environment')
                break
    finally:
        original_error = sys.exc_info()[1]
        try:
            model_client.call(func_name='close_client', obs=tag)
        except Exception:
            if original_error is None:
                raise
            logging.getLogger(__name__).exception('Session cleanup also failed')


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError('Use independent single-episode policy servers')
