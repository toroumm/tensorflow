
from tfdnn import DeepTF
from mysql_connector import Mysql
from mysql_connector import ResultsDatabase
import numpy as np
import sys, os

def setting_and_run(inputs, outputs, teste_in, test_out,path_dnn,base,fold):

    with open(path_dnn, 'r') as f:
        for line in f:
            dnn_conf = line.split()
            if (dnn_conf[0] == '#'):
                break
            else:
                cam = int(dnn_conf[0])
                hidden = np.zeros((cam),dtype=int)
                for i in xrange(0, hidden.shape[0]):
                    hidden[i] = int(dnn_conf[i + 1].replace(',', ''))
                k = cam+1

                dnn = DeepTF()

                dnn.set_learning_rate(float(dnn_conf[k+cam].replace(',', '.')))

                k += cam+1

                dnn.set_momentum(float(dnn_conf[k+cam].replace(',', '.')))

                k+= cam+1

                dnn.set_epochs(int(dnn_conf[k+cam]))

                dnn.set_learning_decay(0)

                dnn.set_batch(50)

                dnn.set_learning_decay_batch(0)

                dnn.set_dropout_threshold(0)

                dnn.set_net_name("TFDNN")

                dnn.build_net(hidden, inputs, outputs)

                dnn.run_tfnet(teste_in,teste_out)

                mysql_results = ResultsDatabase()

                mysql_results.insert_experiments(dnn,base,fold)


path = '/home/jeferson/Dropbox/folds_fmri_cocaina/'

# parametros iniciais



base_ini = 4
base_fim = 4
n_folds = 10
n_testes = 4

for ii in xrange(base_ini, base_fim):

    accuracy_teste = 0
    accuracy_treino = 0

    path_dnn = "/home/jeferson/Dropbox/deep_learning/capitulos_mestrado/dissertacao/capitulos/resultados/script_dnn"

    for jj in xrange(0, n_folds):

        data_in = np.loadtxt(path + 'base_' + str(ii) + '_fmri_treino_fold' + str(jj + 1), dtype=float)

        data_out = np.loadtxt(path + 'base_' + str(ii) + '_fmri_treino_out_fold' + str(jj + 1), dtype=int)

        teste_in = np.loadtxt(path + 'base_' + str(ii) + '_fmri_teste_fold' + str(jj + 1), dtype=float)

        teste_out = np.loadtxt(path + 'base_' + str(ii) + '_fmri_teste_out_fold' + str(jj + 1), dtype=int)

        data_in = (data_in - np.min(data_in) * 1.3) / (np.max(data_in) * 1.3 - np.min(data_in) * 1.3)

        teste_in = (teste_in - np.min(data_in) * 1.3) / (np.max(data_in) * 1.3 - np.min(data_in) * 1.3)

        for kk in xrange(0, n_testes):

            net = setting_and_run(data_in, data_out, teste_in, teste_out,path_dnn,ii,jj)


print "Acabou"




