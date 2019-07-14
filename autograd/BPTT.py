"""
A variety of raw and library rnn implementations using numpy and pytorch, for fun.
This file is just a main driver of sandbox code, and the params apply unequally to each model.

An implementation of BPTT, because I like the math, but could only understand it if i implemented it.

The first crack at this is using discrete inputs and outputs for letter-prediction (26 classes and space).
Inputs are 0/1 reals, and outputs are reals in [0,1] which attempt to learn one-hot 0/1 targets.






auouuod^u^a^^f^fawff^fffffaaaaa<
riewvf^haod^^zfvaxaaaaaaaaaaaaa<
ffas^^^^j^^^f^da^da^^^^^^^ffaaa<
elix^^^^^f^^^^ffffaaaaaaaaaaaaa<
kvfjgfffuooaaaaaaaaaaaaaaaaaaaa<
cbvjjhfauuaiaaaasaaaaa^zit^d^og<
xs^b^jjpff^aaaaaaaaaaaaaaa^aaaa<
sad^^^^^^^^^^^^^j^idauw^^^jazva<
rhj^^^^da^^fasffffffffffaaaaaaa<
auouuuuuuuuuuuuuuuu^as^s^f^aaaa<
bvmjvbvaaaaaaaaaaataoooemmaaaaa<
pffuuu^^^f^a^iaaaa^iaaa^ioaua^^<
cbsfffad^fasua^uauaaeaaeauasaox<
cvffxffsffss^fbaaaaaaaaaaaaaaaa<
 dvsjjzzzaoleeuclaaaaaaaaaaaaaa<
auouuuuuuuuuuuuuuuu^^f^^^f^^^^^<
cbvjfhaamaasaow^azvpaaaaaaaaaaa<
^^bvuuuuuu^^ffmfffmsffffzfavfaa<
wiqvfvamaea^aa^ioa^aoaaaaaaalla<
uwiuuioiosfffffaaaaaaaaaaaaaa^a<
uwff^^fasaffdvfffffaaaaaaaaaaal<
wmmmdjvpamuhuadamaaaaalaaaaaaaa<
qvfffbmbbbbbbjpme^^j^zjjjaoudan<
lvvfffou^^^bbbaaaaaaaallaaolaao<
r^wffjvfao^aaeaaoaaeaolaaalaala<
bmioooeeooegpppppaaaaaaaaaaaaaa<
ffaaa^ia^^^^^^^^^^^^^^^^^^^a^^a<
oooetxmjpbjjjnnosismaaaaaaaaaaa<
hbvffmmffmffffjjffaaaaaaaaaaaaa<
nk^^jjjumud^^ffumfffaaaaaaaaaaa<
Generating 30 sequences with stochastic=False
handtwers huuskulc$$$$$$$$$$$$$<
gut on wash$$$$$$$$$$$$$$$$$$$$<
 cortidh manbers gu$$$$$$$$$$$$<
kuie tivenslo$$$$$$$$$$$$$$$$$$<
lounter sipng$$$$$$$$$$$$$$$$$$<
lounter sipng$$$$$$$$$$$$$$$$$$<
 cortidh manbers gu$$$$$$$$$$$$<
kuie tivenslo$$$$$$$$$$$$$$$$$$<
dlang ther wispl$$$$$$$$$$$$$$$<
s$$t$ capice$inc$$$$$$$$$$$$$$$<
zem w$$$$$$v$$$$$$$$$$$$$$$$$re<
yeeq is nar mris$$$$$$$$$$$$$$$<
kuie tivenslo$$$$$$$$$$$$$$$$$$<
zem w$$$$$$v$$$$$$$$$$$$$$$$$re<
jut ound thery bla$$$$$$$$$$$$$<
kuie tivenslo$$$$$$$$$$$$$$$$$$<
no guarechw wily$$$$$$$$$$$$$$$<
gut on wash$$$$$$$$$$$$$$$$$$$$<
atske wulv$elen$$$$$$$$$$$$$$$$<
 cortidh manbers gu$$$$$$$$$$$$<
atske wulv$elen$$$$$$$$$$$$$$$$<
yeeq is nar mris$$$$$$$$$$$$$$$<
jut ound thery bla$$$$$$$$$$$$$<
 cortidh manbers gu$$$$$$$$$$$$<
lounter sipng$$$$$$$$$$$$$$$$$$<
gut on wash$$$$$$$$$$$$$$$$$$$$<
no guarechw wily$$$$$$$$$$$$$$$<
ithlldul drecl$$$$$$$$$$$$$$$$$<
dlang ther wispl$$$$$$$$$$$$$$$<
no guarechw wily$$$$$$$$$$$$$$$<


Output with a 3-layer hidden layer, and the following parameters:
	python3 BPTT.py  -maxEpochs=5000 -momentum=0.9 -eta=1E-3 --torch-gru -batchSize=5 -numHiddenLayers=3

pqbjje^aoouz^fjaaaaaaaaaaaaaaaa<
 hfzfff^jdsetjmdazjaasjaaaaaaaa<
bwjodjoub^jnujhfvbffssaaaaaaaaa<
vonu^bzdzi^itfiebfptitaaaoohaaa<
mjiue^^^bjtpfpzebuhbgeauaaaaaaa<
utlb^fj^f^f^^f^ze^e^fhwbfihj^fj<
hlmoooo^ffffvjqbvfovfffatfanfaa<
k onvfbwtnf^^bh^u^z^ihfbffaffaa<
sjyjfffffffffffffjfffffffffffvf<
wrffjqbm^jfpejxoazauwiggiaafaaa<
gssjb^^^^^anepewaaaaeaaaaaaaaaa<
atijmpppoabppblffsasfiuuaaaaaaa<
bh^r^fvtb^b^^^lu^^u^^be^bfa^aol<
z bpszde^uqjpjbjbfvjahoveaapaaa<
rwmbbtbbhabjuu^uj^f^jjfaaaaaaaa<
khf^m^^fov^bzff^ff^izcfs^^iewis<
zfbbbhbv^vfbejlovl^uaaaaaaaaaaa<
ne^^boe^ze^bbxfxaefazeeeueecaaa<
zxbcpfmn^j^feqqfiwlanaahaaaaaaa<
cnpmc^cf^lehte^^e^c^fc^fzeheasu<
sqcjfffbt^fb^^ff^^zj^^ff^^^f^f^<
bwyfbj^jo^vbu^uf^j^fffpu^vf^ff^<
oloughfuuuf^ouuvf^vfhbebusaadda<
 ncjffwitijametitaaaaaaaaaaaaaa<
bhfoh^^bhf^z^ff^pxy^ffp^hxz^bfz<
yjmhb^^f^b^p^ie^jb^fjhbphfbffiw<
cffji^fffffjb^fffxj^bfm^fffj^ff<
urmmbmetbfbuexjjifboaaaaaaaaaaa<
uroojjodufqzuouuaaiaaaaaaaaaaaa<
yz^p^bj^^if^bsoew^atjpoxsaaaaaa<
Generating 30 sequences with stochastic=False
^zat aeddeddeshou$$$$$$$$$$$$$$<
atnoi shiredede s$$$$$$$$$$$$$$<
gruilnn iiiiniinniri$$$$$$$$$$$<
du liin wiiingg$$$$$$$$$$$$$$$$<
ceelinne wassdem$$$$$$$$$$$$$$$<
atnoi shiredede s$$$$$$$$$$$$$$<
oktti tiin this$$$$$$$$$$$$$$$$<
id gaanndaddd$$$$$$$$$$$$$$$$$$<
oktti tiin this$$$$$$$$$$$$$$$$<
^zat aeddeddeshou$$$$$$$$$$$$$$<
 qilneded t$$$$$$$$$$$$$$$$$$$$<
atnoi shiredede s$$$$$$$$$$$$$$<
^zat aeddeddeshou$$$$$$$$$$$$$$<
tvort ii isdand$$$$$$$$$$$$$$$$<
id gaanndaddd$$$$$$$$$$$$$$$$$$<
huou thirasherhed$$$$$$$$$$$$$$<
nr iindan then$$$$$$$$$$$$$$$$$<
wxaqn thennndnnd$$$$$$$$$$$$$$$<
huou thirasherhed$$$$$$$$$$$$$$<
oktti tiin this$$$$$$$$$$$$$$$$<
mwha aatiininto$$$$$$$$$$$$$$$$<
yne ian wasaded$$$$$$$$$$$$$$$$<
szden sinnnd hur th$$$$$$$$$$$$<
^zat aeddeddeshou$$$$$$$$$$$$$$<
kthe s iyyen hadde$$$$$$$$$$$$$<
ugni atainggha$$$$$$$$$$$$$$$$$<
xk atannntddeedendendn$$$$$$$$$<
rildatindin thid$$$$$$$$$$$$$$$<
elilthen hou$$$$$$$$$$$$$$$$$$$<
ugni atainggha$$$$$$$$$$$$$$$$$<

Generating 30 sequences with stochastic=False
rerh ifsintishid$$$$$$$$$$$$$$$<
z orretingvksug$$$$$$$$$$$$$$$$<
ster ihed$dad$$$$$$$$$$$$$$$$$$<
lanset$$ wir$$$$$$$$$$$$$$$$$$$<
azrrredy forsin$$$$$$$$$$$$$$$$<
ster ihed$dad$$$$$$$$$$$$$$$$$$<
^^uther whosiny won$$$$$$$$$$$$<
ukdrinsished$$d$$$$$$$$$$$$$$$$<
wzeryontkty$$d$$$$$$$$$$$$$$$$$<
nayathontsai$$$$$$$$$$$$$$$$$$$<
xartyessingluss$$$$$$$$$$$$$$$$<
bzurrn sourked$$$$$$$$$$$$$$$$$<
nayathontsai$$$$$$$$$$$$$$$$$$$<
kalsoudery$$$$$$$$$$$$$$$$$$$$$<
fs k$on$ nad$$$$$$$$$$$$$$$$$$$<
rerh ifsintishid$$$$$$$$$$$$$$$<
xartyessingluss$$$$$$$$$$$$$$$$<
ster ihed$dad$$$$$$$$$$$$$$$$$$<
hazstontud$$d$$$$$$$$$$$$$$$$$$<
card sout$da$$$$$$$$$$$$$$$$$$$<
vquucobdsteron$$$$$$$$$$$$$$$$$<
ukdrinsished$$d$$$$$$$$$$$$$$$$<
y$kesnancled$$$$$$$$$$$$$$$$$$$<
card sout$da$$$$$$$$$$$$$$$$$$$<
d congk$ting$$$$$$$$$$$$$$$$$$$<
 comeded$ her$$$$$$$$$$$$$$$$$$<
$wssweny thirsdr$$$$$$$$$$$$$$$<
tw tradetherdid$$$$$$$$$$$$$$$$<
g$t faavryyend$$$$$$$$$$$$$$$$$<
xartyessingluss$$$$$$$$$$$$$$$$<
"""



import sys
import random
import torch
from numpy_rnn import NumpyRnn
import matplotlib.pyplot as plt
from custom_torch_rnn import *
from torch_gru import *
from data_lib import *

def usage():
	print("Usage: python3 BPTT.py")
	print("Params (these apply differently to selected models): -eta,\
						-maxEpochs/-epochs, \
						-hiddenUnits,\
						-bpStepLimit,\
						-numSequences,\
						-miniBatchSize\
						-maxSeqLen,\
						-clip")
	print("Models: --torch-gru=[rnn or gru], --numpy-rnn, --custom-torch-rnn")
	print("Suggested example params: python3 BPTT.py  -maxEpochs=100000 -momentum=0.9 -eta=1E-3 --torch-gru -batchSize=10 -numHiddenLayers=2")

def main():
	eta = 1E-5
	hiddenUnits = 50
	numHiddenLayers = 1
	maxEpochs = 500
	miniBatchSize = 2
	momentum = 1E-5
	bpStepLimit = 4
	maxSeqLen = 200
	numSequences = 100000
	clip = -1.0 #Only applies to pytorch rnn/gru's, to mitigate exploding gradients, but I don't suggest using it. 
	saveMinWeights = "--saveMinWeights" in sys.argv
	for arg in sys.argv:
		if "-hiddenUnits=" in arg:
			hiddenUnits = int(arg.split("=")[-1])
		if "-eta=" in arg:
			eta = float(arg.split("=")[-1])
		if "-momentum=" in arg:
			momentum = float(arg.split("=")[-1])
		if "-bpStepLimit=" in arg:
			bpStepLimit = int(arg.split("=")[-1])
		if "-maxEpochs=" in arg or "-epochs=" in arg:
			maxEpochs = int(arg.split("=")[-1])
		if "-miniBatchSize=" in arg or "-batchSize" in arg:
			miniBatchSize = int(arg.split("=")[-1])
		if "-numSequences" in arg:
			numSequences = int(arg.split("=")[-1])
		if "-numHiddenLayers=" in arg:
			numHiddenLayers = int(arg.split("=")[-1])
		if "-clip=" in arg:
			clip = float(arg.split("=")[-1])
		if "-maxSeqLen" in arg:
			maxSeqLen = int(arg.split("=")[-1])

	NUMPY_RNN = "--numpy-rnn"
	dataset, encodingMap = BuildCharSequenceDataset(limit=numSequences, maxSeqLen=20)
	reverseEncoding = dict([(encodingMap[key],key) for key in encodingMap.keys()])
	print("Randomizing dataset...")
	random.shuffle(dataset)

	print("First few target outputs:")
	for sequence in dataset[0:20]:
		word = ""
		for x,y in sequence:
			index = np.argmax(y)
			word += reverseEncoding[index]
		print(word)

	print(str(encodingMap))
	#print(str(dataset[0]))
	print("SHAPE: num examples={} xdim={} ydim={}".format(len(dataset), dataset[0][0][0].shape, dataset[0][0][1].shape))
	xDim = dataset[0][0][0].shape[0]
	yDim = dataset[0][0][1].shape[0]

	if "--numpy-rnn" in sys.argv:
		print("TODO: Implement sigmoid and tanh scaling to prevent over-saturation; see Simon Haykin's backprop implementation notes")
		print("TOOD: Implement training/test evaluation methods, beyond the cost function. Evaluate the probability of sequences in train/test data.")
		print("Building rnn with {} inputs, {} hidden units, {} outputs".format(xDim, hiddenUnits, yDim))
		net = NumpyRnn(eta, xDim, hiddenUnits, yDim, lossFunction="SSE", outputActivation="SOFTMAX", hiddenActivation="TANH")
		#train the model
		net.Train(dataset, maxEpochs, miniBatchSize, bpStepLimit=bpStepLimit, clipGrad=clipGrad, momentum=momentum, saveMinWeights=saveMinWeights)
		print("Stochastic sampling: ")
		net.Generate(reverseEncoding, stochastic=True)
		print("Max sampling (expect cycles/repetition): ")
		net.Generate(reverseEncoding, stochastic=False)


if __name__ == "__main__":
	main()

