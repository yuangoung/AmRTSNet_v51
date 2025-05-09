class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'rrhtdata':
            return 'F:\JGS_DATA\AmrtsNET'  # folder that contains rrhtdata/.
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
