


from tfdnn import DeepTF
import numpy as np

#load load files



path_files = '/home/jeferson/MEGA/IEAv/imagens_mosaico/database/lbp/'

i= 16
for i in xrange(16,17, i*2):

    input = np.loadtxt(path_files+str(i) + '_1_8.txt')

    for j in xrange(1,13):

        out = np.loadtxt(path_files+'output_'+str(i) + '_1_8_txt_'+str(j))


        net = DeepTF()

        net.set_batch(1)

        net.set_learning_rate(0.1)

        net.set_show_progress(20)

        hidden = np.array([2000])

        net.build_net(hidden,input,out)

        net.run_tfnet(input,out)