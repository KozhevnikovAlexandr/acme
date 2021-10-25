from acme import specs
import numpy as np
from scapy.all import rdpcap
import dm_env
from datetime import datetime


class MudbusEnv(dm_env.Environment):

    def __init__(self, path, episode_length, input_ip, output_ip, displacement):
        self.input_alphabet, self.output_alphabet, self.inputs, self.outputs = \
            self.get_modbus_data(path, input_ip, output_ip, displacement)
        self.outputs = self.outputs
        self.action_space = len(self.output_alphabet) + 1
        self.observation_space = len(self.input_alphabet)
        self.count = 0
        self.state = self.inputs[0]
        self.episode_length = episode_length
        self.reset_next_step = False
        self.success = 0
        self.timestamp = None
        self.epochs_count = 0
        self.model = None
        self.log = open('log.txt', 'w')

    def test_agent(self):
        succsess_predicts = 0
        end = min(len(self.outputs), len(self.outputs)) - 1
        obs = [self.to_obs(i) for i in range(self.episode_length, 4000)]
        c = 0
        for i in range(len(obs)):
            c += 1
            predict = self.model.select_action(obs[i])
            if predict == self.outputs[self.episode_length + i]:
                succsess_predicts += 1
        print('='*50, file=self.log)
        print('\n ПРОВЕРКА НА ТЕСТОВЫХ ДАННЫХ: {0:.3f}%\n'.format(succsess_predicts / c * 100), file=self.log)
        print('='*50, file=self.log)

    def step(self, action):

        # if self.reset_next_step:
        # return self.reset()
        if self.timestamp == None:
            self.timestamp = datetime.now()
        reward = -1.

        if self.count != self.episode_length:
            if self.outputs[self.count] == action:
                reward = 1.
                self.success += 1
            self.state = self.inputs[self.count]
        else:
            # self.reset_next_step = True
            ep_time = datetime.now() - self.timestamp
            self.epochs_count += 1
            print(
                'Эпоха {0} -- точность {1:.3f}% -- время {2}'.format(self.epochs_count, self.success / self.count * 100,
                                                                     ep_time), file=self.log)
            #print('printed')
            if self.epochs_count % 100 == 0:
                self.test_agent()
            #print('=' * 50)
            self.success = 0
            return dm_env.termination(reward=reward, observation=self.observation())
        self.count += 1
        return dm_env.transition(reward=reward, observation=self.observation())

    def reset_time(self):
        self.timestamp = None
        self.epochs_count = 0

    def to_obs(self, num):
        r = np.zeros(len(self.input_alphabet), dtype='float32')
        r[int(self.inputs[num])] = 1
        return r

    def reset(self):
        self.suck = 0
        self.state = self.inputs[0]
        self.count = 0
        return dm_env.restart(self.observation())

    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(
            dtype='int32', num_values=len(self.output_alphabet), name="action")

    def observation(self) -> np.ndarray:
        obs = np.zeros(len(self.input_alphabet), dtype='float32')
        obs[int(self.inputs[self.count])] = 1
        return obs

    def observation_spec(self):
        return specs.BoundedArray(
            shape=self.observation().shape,
            dtype=self.observation().dtype,
            name="modbus",
            minimum=0,
            maximum=len(self.input_alphabet)
        )

    def reward_spec(self):
        return specs.Array(shape=(), dtype='double', name='reward')

    def discount_spec(self):
        return specs.BoundedArray(shape=(), dtype='double', minimum=0., maximum=1., name='discount')

    def shape(self):
        return self.outputs.shape

    def get_modbus_data(self, path, input_ip, output_ip, displacement):
        pcap = rdpcap(path)
        input_alphabet = dict()
        output_alphabet = dict()
        inputs = []
        outputs = []
        input_count = 0
        output_count = 0
        for i in pcap.res:
            new_data = i['Raw'].load[displacement:]
            if i['IP'].src == input_ip:
                if new_data not in input_alphabet.keys():
                    input_alphabet[new_data] = input_count
                    input_count += 1
                inputs.append(input_alphabet[new_data])

            elif i['IP'].src == output_ip:
                if new_data not in output_alphabet.keys():
                    output_alphabet[new_data] = output_count
                    output_count += 1
                outputs.append(output_alphabet[new_data])

        inputs = np.array(inputs, dtype="float32")
        outputs = np.array(outputs, dtype="int32")
        return input_alphabet, output_alphabet, inputs, outputs
