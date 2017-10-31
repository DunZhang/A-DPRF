package test;

import java.util.Random;
import java.util.Vector;
import modified.weka.classifiers.trees.DiifPRF;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.SMOTE;

public abstract class TestTrees {

	/** 输出该数据集各个类别的实例的数量 */
	private static void testBalance(Instances data) throws Exception {
		int[] count = new int[data.classAttribute().numValues()];
		for (int i = 0; i < data.numInstances(); i++) {
			count[(int) data.instance(i).classValue()]++;
		}
		for (int i = 0; i < count.length; i++)
			System.out.print(String.format("  %d   ", count[i]));
		System.out.println();
	}

	/** 输出正负类的比例 */
	private static double PN(Instances data) throws Exception {
		double[] count = new double[data.classAttribute().numValues()];
		for (int i = 0; i < data.numInstances(); i++) {
			count[(int) data.instance(i).classValue()]++;
		}
		return count[1] / count[0];
	}

	/** 测试数据每个属性含有属性值 */
	private static void testDisc(Instances data) {
		for (int i = 0; i < data.numAttributes(); i++) {
			System.out.print(String.format("  %d   ", data.attribute(i).numValues()));
		}
		System.out.println();
	}

	private static double[] distribution(Instances data) {
		double[] res = new double[data.numClasses()];
		for (int i = 0; i < data.numInstances(); i++) {
			res[(int) data.instance(i).classValue()]++;
		}
		return res;
	}

	/**
	 * 采用留出法获得训练集和测试集
	 * 
	 * @param data
	 *            原始数据
	 * @param percentage
	 *            训练集占数据集的百分比,(0,100)
	 * @param overSamplingOption
	 *            过采样方法 'S':smote方法 null:不进行过采样
	 * @return 数组 0:训练集, 1:测试集
	 */
	private static Instances[] getTrainAndTest_holdOut(Instances data, int percentage, char overSamplingOption,
			double N, long seed) throws Exception {

		Instances train = new Instances(data);
		train.randomize(new Random(seed));
		Instances test = new Instances(train, 0);
		int num = train.numInstances() * (100 - percentage) / 100;
		for (int i = 0; i < num; i++) {
			test.add(train.instance(0));
			train.delete(0);
		}
		// 过采样处理
		if (overSamplingOption == 'S' && PN(train) > 1.0) {
			SMOTE smote = new SMOTE();
			smote.setInputFormat(train);
			if (Double.isNaN(N)) {
				smote.setPercentage((PN(train) - 1) * 100);
			} else {
				smote.setPercentage((PN(train) - 1) / 30 * N * 100);
			}
			train = Filter.useFilter(train, smote);
		}

		// 离散化处理
		Discretize discretize = new Discretize();
		discretize.setInputFormat(train);
		train = Filter.useFilter(train, discretize);
		// 返回结果
		Instances[] res = new Instances[2];
		res[0] = train;
		res[1] = test;
		return res;
	}

	/**
	 * 留出法测试，20次取平均值
	 * 
	 * @param Sampling
	 *            随机森林采样策略 'B':自助法，传统的方法 'A':平均法，每一棵树只能获得总样本的一半
	 * @param overSamplingOption
	 *            过采样方法 'S':smote方法 N:不进行过采样
	 * @param qOption
	 *            指数机制选择时用的评估函数 'I':信息增益 'G':基尼指数 'M':最大类频数
	 * @param seed
	 *            决定20次用的训练集和测试集(20次用的训练集和测试集是不一样的)
	 */

	public static double DPRF_HoldOut(Instances data, int treeNum, int d, double epsilon, char qOption, char Sampling,
			char overSamplingOption, double N1, long seed) throws Exception {

		double sumAuc = 0, sumF1 = 0, sumAcc = 0.0, sumP = 0, sumR = 0;

		data.setClassIndex(data.numAttributes() - 1);
		int seedStep = 0;
		for (int times = 0; times < 20; times++) {

			Instances[] datas = getTrainAndTest_holdOut(data, 70, overSamplingOption, N1, seed + seedStep);
			seedStep++;
			Instances train = datas[0];
			Instances test = datas[1];
			// 检测测试集是否含有0个正类或负类
			double[] tres = distribution(test);
			if (tres[0] == 0 || tres[1] == 0) {
				times--;
				continue;
			}
			DiifPRF rf = new DiifPRF(treeNum);
			rf.buildClassifier_DPRF(train, qOption, d, epsilon, Sampling);
			// AUC
			double T = 0;
			Vector M = new Vector<>();
			Vector N = new Vector<>();
			for (int i = 0; i < test.numInstances(); i++)
				if (test.instance(i).classValue() == 1.0)
					M.add(i);
				else
					N.add(i);

			for (int i = 0; i < M.size(); i++) {
				double pPred = rf.classifyInstance_Prob(test.instance((int) M.get(i)))[1];
				for (int j = 0; j < N.size(); j++) {

					double nPred = rf.classifyInstance_Prob(test.instance((int) N.get(j)))[1];
					if (pPred > nPred)
						T++;
					else if (pPred == nPred)
						T += 0.5;
				}
			}

			double tAUC = (T / (M.size() * N.size()));
			if (tAUC < 0.5)
				tAUC = 1 - tAUC;
			sumAuc += tAUC;

			// Codes for computing F1, if necessary, just cancle the comments

			/*
			 * double TP = 0, FN = 0, FP = 0, TN = 0; for (int i = 0; i <
			 * test.numInstances(); i++) { double truth = test.instance(i).classValue();
			 * double pred = rf.classifyInstance(test.instance(i)); if (pred == 1 && truth
			 * == 1) TP++; else if (pred == 1 && truth == 0) FP++; else if (pred == 0 &&
			 * truth == 1) FN++; else if (pred == 0 && truth == 0) TN++; }
			 * 
			 * sumF1 += (2 * TN) / (2 * TN + FP + FN);
			 */

		}

		return sumAuc / 20;
		// return sumF1 / 20;
	}

	/**
	 * 
	 * @param overSamplingOption
	 *            过采样方法 'S':smote方法 N:不进行过采样
	 * @param qOption
	 *            指数机制选择时用的评估函数 'I':信息增益 'G':基尼指数 'M':最大类频数
	 */
	public static double RF_HoldOut(Instances data, int treeNum, int d, char qOption, char overSamplingOption,
			double N1, long seed) throws Exception {

		double sumAuc = 0, sumF1 = 0, sumAcc = 0.0, sumP = 0, sumR = 0;

		data.setClassIndex(data.numAttributes() - 1);
		int seedStep = 0;
		for (int times = 0; times < 20; times++) {

			Instances[] datas = getTrainAndTest_holdOut(data, 70, overSamplingOption, N1, seed + seedStep);
			seedStep++;
			Instances train = datas[0];
			Instances test = datas[1];
			// 检测测试集是否含有0个正类或负类
			double[] tres = distribution(test);
			if (tres[0] == 0 || tres[1] == 0) {
				times--;
				continue;
			}
			DiifPRF rf = new DiifPRF(treeNum);
			rf.buildClassifier(train, qOption, d);

			// compute AUC

			double T = 0;
			Vector M = new Vector<>();
			Vector N = new Vector<>();
			for (int i = 0; i < test.numInstances(); i++)
				if (test.instance(i).classValue() == 1.0)
					M.add(i);
				else
					N.add(i);

			for (int i = 0; i < M.size(); i++) {
				double pPred = rf.classifyInstance_Prob(test.instance((int) M.get(i)))[1];
				for (int j = 0; j < N.size(); j++) {

					double nPred = rf.classifyInstance_Prob(test.instance((int) N.get(j)))[1];
					if (pPred > nPred)
						T++;
					else if (pPred == nPred)
						T += 0.5;
				}
			}

			double tAUC = (T / (M.size() * N.size()));
			if (tAUC < 0.5)
				tAUC = 1 - tAUC;
			sumAuc += tAUC;

			// Codes for computing F1, if necessary, just cancle the comments

			/*
			 * double TP = 0, FN = 0, FP = 0, TN = 0; for (int i = 0; i <
			 * test.numInstances(); i++) { double truth = test.instance(i).classValue();
			 * double pred = rf.classifyInstance(test.instance(i)); if (pred == 1 && truth
			 * == 1) TP++; else if (pred == 1 && truth == 0) FP++; else if (pred == 0 &&
			 * truth == 1) FN++; else if (pred == 0 && truth == 0) TN++; }
			 * 
			 * sumF1 += (2 * TN) / (2 * TN + FP + FN);
			 */

		}
		return sumAuc / 20;
		// return sumF1 / 20;
	}

	/**
	 * 跨项目预测，20次取平均值
	 * 
	 * @param trainingSet
	 *            原始训练集
	 * @param testSet
	 *            原始测试集合
	 * 
	 * @param Sampling
	 *            随机森林采样策略 'B':自助法，传统的方法 'A':平均法，每一棵树只能获得总样本的一半
	 * @param overSamplingOption
	 *            过采样方法 'S':smote方法 N:不进行过采样
	 * @param qOption
	 *            指数机制选择时用的评估函数 'I':信息增益 'G':基尼指数 'M':最大类频数
	 * @param seed
	 *            决定20次用的训练集和测试集(20次用的训练集和测试集是不一样的)
	 */

	public static double DPRF_CPDP(Instances trainingSet, Instances testSet, int treeNum, int d, double epsilon,
			char qOption, char Sampling, char overSamplingOption, double N1) throws Exception {

		double sumAuc = 0, sumF1 = 0, sumAcc = 0.0, sumP = 0, sumR = 0;
		trainingSet = new Instances(trainingSet);
		testSet = new Instances(testSet);
		trainingSet.setClassIndex(trainingSet.numAttributes() - 1);
		testSet.setClassIndex(testSet.numAttributes() - 1);
		// 过采样处理
		if (overSamplingOption == 'S' && PN(trainingSet) > 1.0) {
			SMOTE smote = new SMOTE();
			smote.setInputFormat(trainingSet);
			if (Double.isNaN(N1)) {
				smote.setPercentage((PN(trainingSet) - 1) * 100);
			} else {
				smote.setPercentage((PN(trainingSet) - 1) / 30 * N1 * 100);
			}
			trainingSet = Filter.useFilter(trainingSet, smote);
		}

		// 离散化处理
		Discretize discretize = new Discretize();
		discretize.setInputFormat(trainingSet);
		trainingSet = Filter.useFilter(trainingSet, discretize);

		for (int times = 0; times < 20; times++) {

			DiifPRF rf = new DiifPRF(treeNum);
			rf.buildClassifier_DPRF(trainingSet, qOption, d, epsilon, Sampling);
			// AUC
			double T = 0;
			Vector M = new Vector<>();
			Vector N = new Vector<>();
			for (int i = 0; i < testSet.numInstances(); i++)
				if (testSet.instance(i).classValue() == 1.0)
					M.add(i);
				else
					N.add(i);

			for (int i = 0; i < M.size(); i++) {
				double pPred = rf.classifyInstance_Prob(testSet.instance((int) M.get(i)))[1];
				for (int j = 0; j < N.size(); j++) {

					double nPred = rf.classifyInstance_Prob(testSet.instance((int) N.get(j)))[1];
					if (pPred > nPred)
						T++;
					else if (pPred == nPred)
						T += 0.5;
				}
			}

			double tAUC = (T / (M.size() * N.size()));
			if (tAUC < 0.5)
				tAUC = 1 - tAUC;
			sumAuc += tAUC;

			// Codes for computing F1, if necessary, just cancle the comments

			// double TP = 0, FN = 0, FP = 0, TN = 0;
			// for (int i = 0; i < testSet.numInstances(); i++) {
			// double truth = testSet.instance(i).classValue();
			// double pred = rf.classifyInstance(testSet.instance(i));
			// if (pred == 1 && truth == 1)
			// TP++;
			// else if (pred == 1 && truth == 0)
			// FP++;
			// else if (pred == 0 && truth == 1)
			// FN++;
			// else if (pred == 0 && truth == 0)
			// TN++;
			// }
			//
			// sumF1 += (2 * TN) / (2 * TN + FP + FN);

		}

		return sumAuc / 20;
		// return sumF1 / 20;
	}

	/**
	 * 
	 * @param overSamplingOption
	 *            过采样方法 'S':smote方法 N:不进行过采样
	 * @param qOption
	 *            指数机制选择时用的评估函数 'I':信息增益 'G':基尼指数 'M':最大类频数
	 */
	public static double RF_CPDP(Instances trainingSet, Instances testSet, int treeNum, int d, char qOption,
			char overSamplingOption, double N1) throws Exception {

		double sumAuc = 0, sumF1 = 0, sumAcc = 0.0, sumP = 0, sumR = 0;
		trainingSet = new Instances(trainingSet);
		testSet = new Instances(testSet);
		trainingSet.setClassIndex(trainingSet.numAttributes() - 1);
		testSet.setClassIndex(testSet.numAttributes() - 1);
		// 过采样处理
		if (overSamplingOption == 'S' && PN(trainingSet) > 1.0) {
			SMOTE smote = new SMOTE();
			smote.setInputFormat(trainingSet);
			if (Double.isNaN(N1)) {
				smote.setPercentage((PN(trainingSet) - 1) * 100);
			} else {
				smote.setPercentage((PN(trainingSet) - 1) / 30 * N1 * 100);
			}
			trainingSet = Filter.useFilter(trainingSet, smote);
		}

		// 离散化处理
		Discretize discretize = new Discretize();
		discretize.setInputFormat(trainingSet);
		trainingSet = Filter.useFilter(trainingSet, discretize);

		for (int times = 0; times < 20; times++) {

			DiifPRF rf = new DiifPRF(treeNum);
			rf.buildClassifier(trainingSet, qOption, d);

			// compute AUC

			double T = 0;
			Vector<Object> M = new Vector<>();
			Vector<Object> N = new Vector<>();
			for (int i = 0; i < testSet.numInstances(); i++)
				if (testSet.instance(i).classValue() == 1.0)
					M.add(i);
				else
					N.add(i);

			for (int i = 0; i < M.size(); i++) {
				double pPred = rf.classifyInstance_Prob(testSet.instance((int) M.get(i)))[1];
				for (int j = 0; j < N.size(); j++) {

					double nPred = rf.classifyInstance_Prob(testSet.instance((int) N.get(j)))[1];
					if (pPred > nPred)
						T++;
					else if (pPred == nPred)
						T += 0.5;
				}
			}

			double tAUC = (T / (M.size() * N.size()));
			if (tAUC < 0.5)
				tAUC = 1 - tAUC;
			sumAuc += tAUC;

			// Codes for computing F1, if necessary, just cancle the comments

			// double TP = 0, FN = 0, FP = 0, TN = 0;
			// for (int i = 0; i < testSet.numInstances(); i++) {
			// double truth = testSet.instance(i).classValue();
			// double pred = rf.classifyInstance(testSet.instance(i));
			// if (pred == 1 && truth == 1)
			// TP++;
			// else if (pred == 1 && truth == 0)
			// FP++;
			// else if (pred == 0 && truth == 1)
			// FN++;
			// else if (pred == 0 && truth == 0)
			// TN++;
			// }
			//
			// sumF1 += (2 * TN) / (2 * TN + FP + FN);

		}
		return sumAuc / 20;
		// return sumF1 / 20;
	}
}
