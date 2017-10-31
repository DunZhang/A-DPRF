package test;

import java.util.Random;
import java.util.Vector;
import modified.weka.classifiers.trees.DiifPRF;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.SMOTE;

public abstract class TestTrees {

	/** ��������ݼ���������ʵ�������� */
	private static void testBalance(Instances data) throws Exception {
		int[] count = new int[data.classAttribute().numValues()];
		for (int i = 0; i < data.numInstances(); i++) {
			count[(int) data.instance(i).classValue()]++;
		}
		for (int i = 0; i < count.length; i++)
			System.out.print(String.format("  %d   ", count[i]));
		System.out.println();
	}

	/** ���������ı��� */
	private static double PN(Instances data) throws Exception {
		double[] count = new double[data.classAttribute().numValues()];
		for (int i = 0; i < data.numInstances(); i++) {
			count[(int) data.instance(i).classValue()]++;
		}
		return count[1] / count[0];
	}

	/** ��������ÿ�����Ժ�������ֵ */
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
	 * �������������ѵ�����Ͳ��Լ�
	 * 
	 * @param data
	 *            ԭʼ����
	 * @param percentage
	 *            ѵ����ռ���ݼ��İٷֱ�,(0,100)
	 * @param overSamplingOption
	 *            ���������� 'S':smote���� null:�����й�����
	 * @return ���� 0:ѵ����, 1:���Լ�
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
		// ����������
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

		// ��ɢ������
		Discretize discretize = new Discretize();
		discretize.setInputFormat(train);
		train = Filter.useFilter(train, discretize);
		// ���ؽ��
		Instances[] res = new Instances[2];
		res[0] = train;
		res[1] = test;
		return res;
	}

	/**
	 * ���������ԣ�20��ȡƽ��ֵ
	 * 
	 * @param Sampling
	 *            ���ɭ�ֲ������� 'B':����������ͳ�ķ��� 'A':ƽ������ÿһ����ֻ�ܻ����������һ��
	 * @param overSamplingOption
	 *            ���������� 'S':smote���� N:�����й�����
	 * @param qOption
	 *            ָ������ѡ��ʱ�õ��������� 'I':��Ϣ���� 'G':����ָ�� 'M':�����Ƶ��
	 * @param seed
	 *            ����20���õ�ѵ�����Ͳ��Լ�(20���õ�ѵ�����Ͳ��Լ��ǲ�һ����)
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
			// �����Լ��Ƿ���0���������
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
	 *            ���������� 'S':smote���� N:�����й�����
	 * @param qOption
	 *            ָ������ѡ��ʱ�õ��������� 'I':��Ϣ���� 'G':����ָ�� 'M':�����Ƶ��
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
			// �����Լ��Ƿ���0���������
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
	 * ����ĿԤ�⣬20��ȡƽ��ֵ
	 * 
	 * @param trainingSet
	 *            ԭʼѵ����
	 * @param testSet
	 *            ԭʼ���Լ���
	 * 
	 * @param Sampling
	 *            ���ɭ�ֲ������� 'B':����������ͳ�ķ��� 'A':ƽ������ÿһ����ֻ�ܻ����������һ��
	 * @param overSamplingOption
	 *            ���������� 'S':smote���� N:�����й�����
	 * @param qOption
	 *            ָ������ѡ��ʱ�õ��������� 'I':��Ϣ���� 'G':����ָ�� 'M':�����Ƶ��
	 * @param seed
	 *            ����20���õ�ѵ�����Ͳ��Լ�(20���õ�ѵ�����Ͳ��Լ��ǲ�һ����)
	 */

	public static double DPRF_CPDP(Instances trainingSet, Instances testSet, int treeNum, int d, double epsilon,
			char qOption, char Sampling, char overSamplingOption, double N1) throws Exception {

		double sumAuc = 0, sumF1 = 0, sumAcc = 0.0, sumP = 0, sumR = 0;
		trainingSet = new Instances(trainingSet);
		testSet = new Instances(testSet);
		trainingSet.setClassIndex(trainingSet.numAttributes() - 1);
		testSet.setClassIndex(testSet.numAttributes() - 1);
		// ����������
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

		// ��ɢ������
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
	 *            ���������� 'S':smote���� N:�����й�����
	 * @param qOption
	 *            ָ������ѡ��ʱ�õ��������� 'I':��Ϣ���� 'G':����ָ�� 'M':�����Ƶ��
	 */
	public static double RF_CPDP(Instances trainingSet, Instances testSet, int treeNum, int d, char qOption,
			char overSamplingOption, double N1) throws Exception {

		double sumAuc = 0, sumF1 = 0, sumAcc = 0.0, sumP = 0, sumR = 0;
		trainingSet = new Instances(trainingSet);
		testSet = new Instances(testSet);
		trainingSet.setClassIndex(trainingSet.numAttributes() - 1);
		testSet.setClassIndex(testSet.numAttributes() - 1);
		// ����������
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

		// ��ɢ������
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
