import os
from subprocess import Popen, PIPE
import glob
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.io import read
import json
import numpy as np

qscript_template='''#!/bin/bash
#$ -N {1}
#$ -pe {2} {3}
#$ -q {4}
#$ -cwd

source ~/.bash_profile

python2 {0}.py 1> {0}.out 2> {0}.err

'''

llnl_qscript_template='''#!/bin/bash
#MSUB -A {3}
#MSUB -l nodes={0}
#MSUB -l walltime={5}
#MSUB -o {1}.o
#MSUB -N {4}
#MSUB -j oe
#MSUB -l partition={2}
#MSUB -q {6}

srun -N{0} -n1 python {1}.py 1> {1}.out 2> {1}.err
'''


# We should update this to automatically retrain with poorly fit test data
amp_template='''from amp import Amp
from ase.io import read
from amp.descriptor.gaussian import Gaussian
from amp.model.{0} import NeuralNetwork

train_images = read('{1}', index=':')

calc = Amp(descriptor=Gaussian(),
           model=NeuralNetwork(hiddenlayers=({2})),
           label='{3}', dblabel='{6}', cores={{'localhost':{5}}})

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

ttplot_template='''
from amp.analysis import plot_parity, plot_error
from ase.io import read
from amp.model import {0}
from amp.utilities import randomize_images
import os

train_images = read('{1}', index=':')
test_images = read('{2}', index=':')

paramfile = '{3}'


fig, train_e_data, train_f_data = plot_error(paramfile,
                                             train_images,
                                             model={0}.NeuralNetwork,
                                             label='train-error',
                                             plot_forces={4},
                                             dblabel='{5}',
                                             overwrite=True,
                                             returndata=True)

fig, test_e_data, test_f_data = plot_error(paramfile,
                                           test_images,
                                           model={0}.NeuralNetwork,
                                           label='test-error',
                                           plot_forces={4},
                                           dblabel='{5}',
                                           overwrite=True,
                                           returndata=True)

plot_parity(paramfile,
            train_images,
            model={0}.NeuralNetwork,
            label='train-parity',
            plot_forces={4},
            dblabel='{5}',
            overwrite=True,
            returndata=False)

plot_parity(paramfile,
            test_images,
            model={0}.NeuralNetwork,
            label='test-parity',
            plot_forces={4},
            dblabel='{5}',
            overwrite=True,
            returndata=False)

import json

with open('train-error-data.json', 'wb') as f:
    json.dump(train_e_data, f)

with open('test-error-data.json', 'wb') as f:
    json.dump(test_e_data, f)
'''

tvtplot_template='''
from amp.analysis import plot_parity, plot_error
from ase.io import read
from amp.model import {0}
from amp.utilities import randomize_images
import os

train_images = read('{1}', index=':')
validate_images = read('{2}', index=':')
test_images = read('{3}', index=':')

paramfile = '{4}'

fig, train_e_data, train_f_data = plot_error(paramfile,
                                             train_images,
                                             model={0}.NeuralNetwork,
                                             label='train-error',
                                             plot_forces={5},
                                             dblabel='{6}',
                                             overwrite=True,
                                             returndata=True)

fig, valid_e_data, valid_f_data = plot_error(paramfile,
                                             validate_images,
                                             model={0}.NeuralNetwork,
                                             label='validate-error',
                                             plot_forces={5},
                                             dblabel='{6}',
                                             overwrite=True,
                                             returndata=True)


fig, test_e_data, test_f_data = plot_error(paramfile,
                                           test_images,
                                           model={0}.NeuralNetwork,
                                           label='test-error',
                                           plot_forces={5},
                                           dblabel='{6}',
                                           overwrite=True,
                                           returndata=True)

plot_parity(paramfile,
            train_images,
            model={0}.NeuralNetwork,
            label='train-parity',
            plot_forces={5},
            dblabel='{6}',
            overwrite=True,
            returndata=False)

plot_parity(paramfile,
            validate_images,
            model={0}.NeuralNetwork,
            label='validate-parity',
            plot_forces={5},
            dblabel='{6}',
            overwrite=True,
            returndata=False)

plot_parity(paramfile,
            test_images,
            model={0}.NeuralNetwork,
            label='test-parity',
            plot_forces={5},
            dblabel='{6}',
            overwrite=True,
            returndata=False)

import json

with open('train-error-data.json', 'wb') as f:
    json.dump(train_e_data, f)

with open('validate-error-data.json', 'wb') as f:
    json.dump(valid_e_data, f)

with open('test-error-data.json', 'wb') as f:
    json.dump(test_e_data, f)

get_rmse("./")
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
            f.write('python2 {0}.py 1> {0}.out 2> {0}.err'.format(rscript))
    return


def write_llnl_qscript(qscript='qscript',
                       pyscript='amp-script',
                       jobname='amp',
                       partition='quartz',
                       nodes=1,
                       bank='ioncond',
                       q='pbatch',
                       walltime='24:00:00',
                       restart=True,
                       rscript='amp-restart'):  

    with open(qscript, 'w') as f:
        f.write(llnl_qscript_template.format(nodes,
                                             pyscript,
                                             partition,
                                             bank,
                                             jobname,
                                             walltime,
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
                     dblabel='training',
                     name='amp-script',
                     archive=False, cores=24):
    with open('{0}.py'.format(name), 'w') as f:
        f.write(amp_template.format(model,
                                    traindb,
                                    hl,
                                    label,
                                    archive, cores, dblabel))
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

    with open(logfile) as f:
        lines = f.readlines()
        ermse = float(lines[-4].split()[3])
        frmse = float(lines[-4].split()[-4])
    return ermse, frmse


def make_parity_error_plots(trainfile,
                            testfile,
                            paramfile='rtraining-untrained-parameters.amp',
                            dblabel='training',
                            plot_forces=True,
                            submit=True,
                            model='tflow',
                            name='pplots'):
    
    with open('{0}.py'.format(name), 'w') as f:
        f.write(ttplot_template.format(model,
                                     trainfile,
                                     testfile,
                                     paramfile,
                                     plot_forces,
                                     dblabel))

    if submit:
        write_qscript(qscript='pplots-qscript',
                      pyscript='pplots',
                      jobname='amp-pplots',
                      pe='smp',
                      ncores=24,
                      q='*@@schneider_d12chas',
                      restart=False)
        run_job('pplots-qscript')


def make_tvt_parity_error_plots(trainfile,
                                validationfile,
                                testfile,
                                paramfile='rtraining-untrained-parameters.amp',
                                dblabel='training',
                                plot_forces=True,
                                submit=True,
                                model='tflow',
                                name='tvtpplots'):
    
    with open('{0}.py'.format(name), 'w') as f:
        f.write(tvtplot_template.format(model,
                                        trainfile,
                                        validationfile,
                                        testfile,
                                        paramfile,
                                        plot_forces,
                                        dblabel))

    if submit:
        write_qscript(qscript='tvtpplots-qscript',
                      pyscript='tvtpplots',
                      jobname='amp-tvtpplots',
                      pe='smp',
                      ncores=24,
                      q='*@@schneider_d12chas',
                      restart=False)
        run_job('tvtpplots-qscript')


def vasptraj2db(db, directory):

    # traj = get_traj(calc=calc)
    dir_name = directory.split('/')[-1]
    vasprun = '{0}/vasprun.xml'.format(directory)
    for i, atoms in enumerate(read(vasprun, ':')):
        atoms.set_constraint(None)
        atoms.set_calculator(SPC(atoms,
                                 energy=atoms.get_potential_energy(),
                                 forces=atoms.get_forces()))
        db.write(atoms, image_no=int(i), path=directory)
    return


def write_full_db(db, rootdir):
    from jasp import utils as ju
    dirs = ju.get_jasp_dirs(rootdir)
    for d in dirs:
        vasptraj2db(db, d)


def write_train_test_db(dbpath, dbname='all_images.db', fraction=0.8):
    from amp.utilities import randomize_images
    from ase.db import connect
    
    all_atoms=read('{0}/{1}'.format(dbpath, dbname), ':')
    train, test = randomize_images(all_atoms, fraction=fraction)

    db_train = '{0}/train-{1}.db'.format(dbpath, fraction)
    db_test = '{0}/test-{1}.db'.format(dbpath, 1 - fraction)

    with connect(db_train) as db:
	for i, im in enumerate(train):
	    db.write(im)

    with connect(db_test) as db:
	for i, im in enumerate(test):
	    db.write(im)


def get_rmse(directory="./", validate=False):

    """
    Compute the training and testing rmse from 
    train-error-data.json and test-error-data.json
    """
    
    with cd(directory):
        with open('test-error-data.json') as f:
            test_d = json.load(f)
        with open('train-error-data.json') as f:
            train_d = json.load(f)

        test_error = np.array([test_d[key][-1] for key in test_d.keys()])
        test_rmse = np.sqrt(sum(test_error ** 2) / len(test_error))

        train_error = np.array([train_d[key][-1] for key in train_d.keys()])
        train_rmse = np.sqrt(sum(train_error ** 2) / len(train_error))

        if validate:
            validate_error = np.array([train_d[key][-1] for key in train_d.keys()])
            validate_rmse = np.sqrt(sum(train_error ** 2) / len(train_error))

        if not validate:
            with open('tt-rmse.txt', 'w') as f:
                f.write('{0}\t{1}'.format(train_rmse, test_rmse))
            return train_rmse, test_rmse
        else:
            with open('tvt-rmse.txt', 'w') as f:
                f.write('{0}\t{1}\t{2}'.format(train_rmse, validate_rmse, test_rmse))                
            return train_rmse, validate_rmse, test_rmse


energy_eval_template='''
import numpy as np
import os
from amp import Amp
from amp.model import tflow
from amp.utilities import hash_images
from ase.db import connect
from ase.io import read

images = read('{4}', ':')
os.chdir('{0}')
calc = Amp.load('{1}', Model=tflow.NeuralNetwork, dblabel='{2}', label='{3}')

images = hash_images(images, log=calc.log)

calc.descriptor.calculate_fingerprints(images = images,
                                       cores = calc.cores,
                                       log=calc.log,
                                       calculate_derivatives=False)

energies, force = calc.model.get_energy_list(images.keys(), calc.descriptor.fingerprints)

with open('{3}.npy', 'wb') as f:
    np.save(f, energies)
'''


def eval_energies(grid_dbpath,
                  calc_directory='./',
                  paramfile='rtraining-untrained-parameters.amp',
                  dblabel='training',
                  label='energy_eval',
                  submit=True):

    with open('{0}.py'.format(label), 'w') as f:
        f.write(energy_eval_template.format(calc_directory,
                                            paramfile,
                                            dblabel,
                                            label,
                                            grid_dbpath))

    if submit:
        write_qscript(qscript='{0}-qscript'.format(label),
                      pyscript=label,
                      jobname=label,
                      pe='smp',
                      ncores=24,
                      q='*@@schneider_d12chas',
                      restart=False)
        run_job('{0}-qscript'.format(label))
