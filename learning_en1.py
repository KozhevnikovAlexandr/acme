from acme import specs
import sonnet as snt
import acme.tf.networks as networks
import acme.agents.tf.r2d2 as r2d2
from acme.environment_loop import EnvironmentLoop
from acme.utils.loggers import InMemoryLogger
from Envs import MudbusEnv


class SimpleNetwork(networks.RNNCore):

    def __init__(self, action_spec):
        super().__init__(name='r2d2_test_network')
        self._net = snt.DeepRNN([
            snt.LSTM(327),
            snt.Flatten(),
            snt.Linear(13)
        ])

    def __call__(self, inputs, state):
        return self._net(inputs, state)

    def initial_state(self, batch_size: int, **kwargs):
        return self._net.initial_state(batch_size)

    def unroll(self, inputs, state, sequence_length):
        return snt.static_unroll(self._net, inputs, state, sequence_length)


if __name__ == '__main__':
    env = MudbusEnv('Datasets/modbus_clever_office_131-218_full_bad(2).pcap', 3000, '192.168.12.131', '192.168.252.218', 6)
    environment_spec = specs.make_environment_spec(env)
    agent = r2d2.R2D2(
        environment_spec=environment_spec,
        network=SimpleNetwork(env),
        batch_size=64,  # smaller possible but bad, bigger (256) super bad
        samples_per_insert=32,
        min_replay_size=100,
        store_lstm_state=True,
        burn_in_length=4,  # super sensible
        trace_length=5,  # sensible smaller bad bigger too
        replay_period=4,
        checkpoint=True,
        # learning rate has to be lowered to avoid jumping|
        learning_rate=1e-4,
    )
    env.model = agent
    env.reset_time()
    loop = EnvironmentLoop(env, agent, logger=InMemoryLogger())
    env.reset()
    loop.run(5000)
