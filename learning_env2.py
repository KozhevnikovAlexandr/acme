from acme import specs
import sonnet as snt
import acme.tf.networks as networks
import acme.agents.tf.r2d2 as r2d2
from acme.environment_loop import EnvironmentLoop
from acme.utils.loggers import InMemoryLogger
from Envs import MudbusEnv

class SimpleNetwork2(networks.RNNCore):

    def __init__(self, action_spec):
        super().__init__(name='r2d2_test_network')
        self._net = snt.DeepRNN([
            snt.LSTM(20),
            #snt.Flatten(),
            snt.Linear(516)
            ])

    def __call__(self, inputs, state):
        return self._net(inputs, state)

    def initial_state(self, batch_size: int, **kwargs):
        return self._net.initial_state(batch_size)

    def unroll(self, inputs, state, sequence_length):
        return snt.static_unroll(self._net, inputs, state, sequence_length)

if __name__=='__main__':
    env = MudbusEnv('Datasets/modbus_github_234_85.pcap', 12000, '10.1.1.234', '10.10.5.85', 6)
    environment_spec = specs.make_environment_spec(env)
    network = SimpleNetwork2(env)
    agent = r2d2.R2D2(
                environment_spec=environment_spec,
                network=network,
                batch_size=64, # smaller possible but bad, bigger (256) super bad
                samples_per_insert=32,
                min_replay_size=100,
                store_lstm_state=True,
                burn_in_length=5, # super sensible
                trace_length=50, # sensible smaller bad bigger too
                replay_period=4,
                checkpoint=True,
                # learning rate has to be lowered to avoid jumping|
                learning_rate=1e-4,
        )
    env.model = agent
    env.reset_time()
    loop = EnvironmentLoop(env, agent, logger=InMemoryLogger())
    env.reset()
    loop.run(7500)
