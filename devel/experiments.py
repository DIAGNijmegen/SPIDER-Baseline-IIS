import shutil
from os import path, makedirs
from collections import defaultdict, OrderedDict
from glob import glob

from tiger.workflow import ExperimentSettings

import config


class Experiment(ExperimentSettings):
    def __init__(self):
        super().__init__(config.expdir)
        self.add_param('version', default=config.version, cmdarg=False)

    def store_dataset_details(self, dataset):
        # Generate an object containing all datasets, subsets and scans
        subsets = defaultdict(list)

        for subset in dataset.subsets:
            for d, s in dataset.items(subset):
                subsets[subset].append('{}_{}'.format(d.identifier, s.identifier))

        # Sort by subset name
        subsets = OrderedDict(sorted(subsets.items(), key=lambda item: item[0]))

        # Write list of scans to JSON file
        self.dump_into_json_file(subsets, path.join(self.folder, 'datasets.json'))

    def epochs(self):
        files = glob(path.join(self.folder, 'weights', '*.pkl'))
        return [int(path.basename(f)[:-4]) for f in sorted(files)]

    def network_parameter_file(self, epoch):
        return path.join(self.folder, 'weights', '{:06d}.pkl'.format(epoch))

    def learning_curve_file(self, name):
        return path.join(self.folder, 'learning_curves', '{}.csv'.format(name))

    def screenshot_file(self, phase, epoch):
        return path.join(self.folder, 'screenshots', '{}_{:06d}.jpg'.format(phase, epoch))

    def results_folder(self, dataset, ensure_exists=True, name='results'):
        folder = path.join(self.folder, name, dataset.identifier)
        if ensure_exists and not path.exists(folder):
            makedirs(folder)
        return folder

    def trails_folder(self, dataset, scan, clear):
        folder = path.join(self.folder, 'trails', dataset.identifier, scan.identifier)

        if clear:
            # Clear folder and dump arguments into json file
            if path.exists(folder):
                shutil.rmtree(folder)
            makedirs(folder)
            self.dump_arguments(path.join(folder, 'arguments.json'))

        return folder
