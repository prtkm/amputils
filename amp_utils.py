from subprocess import Popen, PIPE
import glob

qscript_template='''#!/bin/bash
#$ -N {1}
#$ -pe {2} {3}
#$ -q {4}
#$ -cwd

source ~/.bash_profile

python {0}.py 1> {0}.out 2> {0}.err

'''

# We should update this to automatically retrain with poorly fit test data
amp_template='''from amp import Amp
from ase.io import read
from amp.descriptor.gaussian import Gaussian
from amp.model.{0} import NeuralNetwork

train_images = read('{1}', index=':')

calc = Amp(descriptor=Gaussian(),
           model=NeuralNetwork(hiddenlayers=({2})),
           label='{3}')

calc.train(images=train_images)

archive = {4}
if archive:
    from amp_utils import archive_dbs
    archive_dbs()
'''

restart_template='''from amp import Amp
from ase.io import read
from amp.descriptor.gaussian import Gaussian
from amp.model.{0} import NeuralNetwork

try:
    calc = Amp.load('{2}-untrained-parameters.amp', 
                    Model=NeuralNetwork, 
                    label='r{2}',
                    dblabel='{3}')
    train_images = read('{1}', index=':')
    calc.train(images=train_images)
except:
    pass

archive = {4}
if archive:
    from amp_utils import archive_dbs
    archive_dbs()
'''

class cd:

    '''Context manager for changing directories.
    On entering, store initial location, change to the desired directory,
    creating it if needed.  On exit, change back to the original directory.
    '''

    def __init__(self, working_directory):
        self.origin = os.getcwd()
        self.wd = working_directory

    def __enter__(self):
        # make directory if it doesn't already exist
        if not os.path.isdir(self.wd):
            os.makedirs(self.wd)

        # now change to new working dir
        os.chdir(self.wd)

    def __exit__(self, *args):
        os.chdir(self.origin)
        return False  # allows body exceptions to propagate out.


def write_qscript(qscript='qscript',
                  pyscript='amp-script',
                  jobname='amp',
                  pe='smp',
                  ncores=24,
                  q='*@@schneider_d12chas',
                  restart=True,
                  rscript='amp-restart'):  
    
    with open(qscript, 'w') as f:
        f.write(qscript_template.format(pyscript,
                                        jobname,
                                        pe,
                                        ncores,
                                        q))
        if restart:
            f.write('python {0}.py 1> {0}.out 2> {0}.err'.format(rscript))
    return


def run_job(qscript='qscript'):

    p = Popen(['qsub', qscript], stdin=PIPE, stdout=PIPE, stderr=PIPE)
                
    out, err = p.communicate()
    jobid = None
    if out == '' or err != '':
        print "hello"
        raise Exception('something went wrong in qsub:\n\n{0}'.format(err))

        jobid = out.split()[2]

        with open('jobid', 'w') as f:
            f.write(jobid) 
    return jobid


def write_amp_script(traindb,
                     model='tflow',
                     hl='10, 10',
                     label='training',
                     name='amp-script',
                     archive=False):
    with open('{0}.py'.format(name), 'w') as f:
        f.write(amp_template.format(model,
                                    traindb,
                                    hl,
                                    label,
                                    archive))
    return


def write_restart_script(traindb,
                         model='tflow',
                         label='training',
                         dblabel='training',
                         name='amp-restart',
                         archive=True):

    with open('{0}.py'.format(name), 'w') as f:
        f.write(restart_template.format(model,
                                        traindb,
                                        label,
                                        dblabel,
                                        archive))
    return


def archive_dbs(directory='./'):
    
    from amp.utilities import FileDatabase
    with cd(directory):
        for filename in glob.glob('*.ampdb'):
            db = FileDatabase(filename)
            db.archive()


def get_training_rmse(logfile):
    """
    Get energy and force rmse from log file
    Usually the params are "unconverged". May need to change
    line index if converged
    """

    with open(log) as f:
        lines = f.readlines()
        ermse = float(lines[-4].split()[3])
        frmse = float(lines[-4].split()[-4])
    return ermse, frmse
