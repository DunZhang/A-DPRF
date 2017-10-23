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
	 * 基于ID3的随机森林
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
	 * 基于DiffPID3_MAX的随机森林
	 * 
	 * @param data
	 *            训练集
	 * @param Sampling
	 *            随机森林采样策略 'B':自助法，传统的方法 'A':平均法，每一棵树只能获得总样本的一半
	 * @param qOption
	 *            指数机制选择时用的评估函数 'I':信息增益 'G':基尼指数 'M':最大类频数
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

		// 自助采样
		if (Sampling == 'B') {
			for (int i = 0; i < numTrees; i++) {
				Instances tdata = new Instances(dataTrain, 0);
				Random r1 = new Random();
				for (int j = 0; j < NumIns; j++) {
					tdata.add(dataTrain.instance(r1.nextInt(NumIns)));
				}
				id3s[i].DiffPID3BuildClassifier(tdata, null, qOption, d, epsilon / numTrees);
			}

		} else if (Sampling == 'A') {// 一次采一半
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
	 * 预测结果
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
	 * 预测为每个类别的概率值
	 */
	public double[] classifyInstance_Prob(Instance instance) throws Exception {

		double[] res = new double[instance.classAttribute().numValues()];

		for (int i = 0; i < id3s.length; i++) {
			res[(int) (id3s[i].classifyInstance(instance))]++;
		}
		// 归一化
		for (int i = 0; i < res.length; i++)
			res[i] /= numTrees;
		return res;
	}
}
