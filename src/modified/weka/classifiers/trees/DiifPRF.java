package modified.weka.classifiers.trees;

import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class DiifPRF {
	private DiffPID3[] id3s;
	private int numTrees;

	public DiifPRF(int numTrees) {
		this.numTrees = numTrees;
		id3s = new DiffPID3[numTrees];
	}

	private Instances[] binaryPart(Instances data) {
		data.randomize(new Random());
		Instances[] datas = new Instances[2];
		datas[0] = new Instances(data, 0);
		datas[1] = new Instances(data, 0);
		for (int i = 0; i < data.numInstances() - 1; i += 2) {
			datas[0].add(data.instance(i));
			datas[1].add(data.instance(i + 1));
		}
		if (data.numInstances() % 2 == 1)
			datas[new Random().nextInt(2)].add(data.instance(data.numInstances() - 1));

		int num1 = datas[0].numInstances();
		Random random = new Random();
		for (int i = 0; i < num1; i++) {
			datas[0].add(datas[0].instance(random.nextInt(num1)));
		}
		int num2 = datas[1].numInstances();
		random = new Random();
		for (int i = 0; i < num2; i++) {
			datas[1].add(datas[1].instance(random.nextInt(num2)));
		}

		return datas;
	}

	/**
	 * ����ID3�����ɭ��
	 */
	public void buildClassifier(Instances data, char qOption, int d) throws Exception {
		Instances dataTrain = new Instances(data);
		dataTrain.randomize(new Random());
		int NumIns = data.numInstances();

		for (int i = 0; i < numTrees; i++) {
			id3s[i] = new DiffPID3();

			// create a new random data set
			Instances tdata = new Instances(dataTrain, 0);

			Random r1 = new Random();
			for (int j = 0; j < NumIns; j++) {
				tdata.add(dataTrain.instance(r1.nextInt(NumIns)));
			}
			id3s[i].buildClassifier(tdata, null, qOption, d);
		}
	}

	/**
	 * ����DiffPID3_MAX�����ɭ��
	 * 
	 * @param data
	 *            ѵ����
	 * @param Sampling
	 *            ���ɭ�ֲ������� 'B':����������ͳ�ķ��� 'A':ƽ������ÿһ����ֻ�ܻ����������һ��
	 * @param qOption
	 *            ָ������ѡ��ʱ�õ��������� 'I':��Ϣ���� 'G':����ָ�� 'M':�����Ƶ��
	 */
	public void buildClassifier_DPRF(Instances data, char qOption, int d, double epsilon, char Sampling)
			throws Exception {
		Instances dataTrain = new Instances(data);
		Random random = new Random();
		dataTrain.randomize(random);
		int NumIns = data.numInstances();
		for (int i = 0; i < numTrees; i++) {
			id3s[i] = new DiffPID3();
		}

		// ��������
		if (Sampling == 'B') {
			for (int i = 0; i < numTrees; i++) {
				Instances tdata = new Instances(dataTrain, 0);
				Random r1 = new Random();
				for (int j = 0; j < NumIns; j++) {
					tdata.add(dataTrain.instance(r1.nextInt(NumIns)));
				}
				id3s[i].DiffPID3BuildClassifier(tdata, null, qOption, d, epsilon / numTrees);
			}

		} else if (Sampling == 'A') {// һ�β�һ��
			if (numTrees % 2 == 0) {
				for (int i = 0; i < numTrees; i += 2) {
					Instances[] tdata = binaryPart(dataTrain);
					id3s[i].DiffPID3BuildClassifier(tdata[0], null, qOption, d, (epsilon / numTrees )* 2);
					id3s[i + 1].DiffPID3BuildClassifier(tdata[1], null, qOption, d, (epsilon / numTrees) * 2);
				}
			} else {
				for (int i = 0; i < numTrees - 1; i += 2) {
					Instances[] tdata = binaryPart(dataTrain);
					id3s[i].DiffPID3BuildClassifier(tdata[0], null, qOption, d, epsilon / (numTrees + 1) * 2);
					id3s[i + 1].DiffPID3BuildClassifier(tdata[1], null, qOption, d, epsilon / (numTrees + 1) * 2);
				}
				id3s[numTrees - 1].DiffPID3BuildClassifier(dataTrain, null, qOption, d, epsilon / (numTrees + 1) * 2);
			}
		}
	}

	/**
	 * Ԥ����
	 */
	public double classifyInstance(Instance instance) throws Exception {

		int[] res = new int[instance.classAttribute().numValues()];
		for (int i = 0; i < id3s.length; i++) {
			res[(int) (id3s[i].classifyInstance(instance))]++;
		}
		if (res.length == 2 && res[0] == res[1])
			return new Random().nextInt(2);
		else
			return Utils.maxIndex(res);

	}

	/**
	 * Ԥ��Ϊÿ�����ĸ���ֵ
	 */
	public double[] classifyInstance_Prob(Instance instance) throws Exception {

		double[] res = new double[instance.classAttribute().numValues()];

		for (int i = 0; i < id3s.length; i++) {
			res[(int) (id3s[i].classifyInstance(instance))]++;
		}
		// ��һ��
		for (int i = 0; i < res.length; i++)
			res[i] /= numTrees;
		return res;
	}
}
