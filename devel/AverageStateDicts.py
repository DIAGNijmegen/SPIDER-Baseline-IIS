import torch

from collections import OrderedDict

from experiments import Experiment


class Module(Experiment):
    def __init__(self, argv):
        super().__init__()
        self.add_param('from_epoch', type=int)
        self.add_param('to_epoch', type=int)
        self.add_param('step', type=int, default=0)
        self.add_param('make_epoch', type=int)
        self.parse_and_restore(argv, execute_preserved_code=False)

    def execute(self):
        # Load all state dicts from snapshots
        print('Loading state dicts...')
        step = self['step'] if self['step'] > 0 else self['snapshotevery']
        state_dicts = [
            torch.load(self.network_parameter_file(epoch))
            for epoch in range(self['from_epoch'], self['to_epoch'] + 1, step)
        ]
        print(f' > {len(state_dicts)} state dicts loaded')

        # Average parameter values
        print('Averaging state dicts...')
        averaged_state_dict = dict()

        for component in ('network',):
            averaged_state_dict[component] = OrderedDict()
            for param in state_dicts[0][component]:
                print(f' > {component}/{param}')
                averaged_state_dict[component][param] = torch.stack([
                    state_dict[component][param] for state_dict in state_dicts
                ], dim=0).mean(dim=0)

        # Store averaged state dicts
        print('Saving new state dict...')
        averaged_state_dict['epochs'] = {'from': self['from_epoch'], 'to': self['to_epoch']}
        torch.save(averaged_state_dict, self.network_parameter_file(self['make_epoch']))

        print('Done')
