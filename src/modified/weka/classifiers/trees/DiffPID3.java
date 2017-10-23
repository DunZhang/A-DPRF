package modified.weka.classifiers.trees;

import java.util.Enumeration;
import java.util.Random;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.UnsupportedAttributeTypeException;
import weka.core.UnsupportedClassTypeException;
import weka.core.Utils;

/**
 * Class implementing an Id3 decision tree classifier. For more information, see
 * <p>
 *
 * R. Quinlan (1986). <i>Induction of decision trees</i>. Machine Learning.
 * Vol.1, No.1, pp. 81-106.
 * <p>
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 1.14.2.1 $
 */
public class DiffPID3 {

	/** The node's successors. */
	private DiffPID3[] m_Successors;

	/** Attribute used for splitting. */
	private Attribute m_Attribute;

	/** Class value if node is leaf. */
	private double m_ClassValue;

	/** Class distribution if node is leaf. */
	private double[] m_Distribution;

	/** Class attribute of dataset. */
	private Attribute m_ClassAttribute;

	/* 该节点的分裂属性在本质上是不是连续型的 */
	private boolean m_numeric;

	/** 连续属性的多个分裂点 */
	public double[] m_cutPoints;

	/** 获取实例集中类别的分布 */
	private double[] classDistribution(Instances data) throws Exception {
		double[] count = new double[data.classAttribute().numValues()];
		for (int i = 0; i < data.numInstances(); i++)
			count[(int) data.instance(i).classValue()]++;
		return count;
	}

	/** 判断本质是不是连续属性 */
	public boolean originNumeric(Instances data, int attrIndex) throws Exception {
		if (data.attribute(attrIndex).isNumeric() || data.attribute(attrIndex).value(0).contains("-inf")
				|| data.attribute(attrIndex).value(0).contains("All"))
			return true;
		else
			return false;
	}

	/** 从预离散化的连续属性获取分裂值 */
	public void getCutPoints(Instances data, int attrIndex) throws Exception {
		for (int i = 0; i < data.attribute(attrIndex).numValues() - 1; i++) {
			String value = data.attribute(attrIndex).value(i);
			int startIndex = 0, endIndex = 0;
			if (i == 0) {
				startIndex = 7;
				endIndex = value.length() - 2;
			} else {
				startIndex = value.indexOf("-") + 1;
				endIndex = value.length() - 2;
			}
			double ivalue = Double.valueOf(value.substring(startIndex, endIndex));
			m_cutPoints[i] = ivalue;
		}
	}

	/**
	 * Classifies a given test instance using the decision tree.
	 *
	 * @param instance
	 *            the instance to be classified
	 * @return the classification
	 */
	public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {

		if (instance.hasMissingValue()) {
			throw new NoSupportForMissingValuesException(" no missing values, " + "please.");
		}

		if (m_Attribute == null) {
			return m_ClassValue;
		} else {
			if (!m_numeric) {
				return m_Successors[(int) instance.value(m_Attribute)].classifyInstance(instance);

			} else {
				int index = 0;
				double val = instance.value(m_Attribute);
				for (; index < m_cutPoints.length; index++) {
					if (val < m_cutPoints[index])
						break;
				}
				return m_Successors[index].classifyInstance(instance);
			}
		}
	}

	/**
	 * Computes class distribution for instance using decision tree.
	 *
	 * @param instance
	 *            the instance for which distribution is to be computed
	 * @return the class distribution for the given instance
	 */
	public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {

		if (instance.hasMissingValue()) {
			throw new NoSupportForMissingValuesException("no missing values, " + "please.");
		}
		if (m_Attribute == null) {
			return m_Distribution;
		} else {
			return m_Successors[(int) instance.value(m_Attribute)].distributionForInstance(instance);
		}
	}

	/**
	 * Computes information gain for an attribute.
	 *
	 * @param data
	 *            the data for which info gain is to be computed
	 * @param att
	 *            the attribute
	 * @return the information gain for the given attribute and data
	 */
	private double computeInfoGain(Instances data, Attribute att) throws Exception {

		double infoGain = computeEntropy(data);
		Instances[] splitData = splitData(data, att);
		for (int j = 0; j < att.numValues(); j++) {
			if (splitData[j].numInstances() > 0) {
				infoGain -= ((double) splitData[j].numInstances() / (double) data.numInstances())
						* computeEntropy(splitData[j]);
			}
		}
		return infoGain;
	}

	/**
	 * Computes the entropy of a dataset.
	 * 
	 * @param data
	 *            the data for which entropy is to be computed
	 * @return the entropy of the data's class distribution
	 */
	private double computeEntropy(Instances data) throws Exception {

		double[] classCounts = new double[data.numClasses()];
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			classCounts[(int) inst.classValue()]++;
		}
		double entropy = 0;
		for (int j = 0; j < data.numClasses(); j++) {
			if (classCounts[j] > 0) {
				entropy -= classCounts[j] * Utils.log2(classCounts[j]);
			}
		}
		entropy /= data.numInstances();
		return entropy + Utils.log2(data.numInstances());
	}

	/**
	 * Splits a dataset according to the values of a nominal attribute.
	 *
	 * @param data
	 *            the data which is to be split
	 * @param att
	 *            the attribute to be used for splitting
	 * @return the sets of instances produced by the split
	 */
	private Instances[] splitData(Instances data, Attribute att) {

		Instances[] splitData = new Instances[att.numValues()];
		for (int j = 0; j < att.numValues(); j++) {
			splitData[j] = new Instances(data, data.numInstances());
		}
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			splitData[(int) inst.value(att)].add(inst);
		}
		for (int i = 0; i < splitData.length; i++) {
			splitData[i].compactify();
		}
		return splitData;
	}

	private double sgn(double x) {
		if (x > 0)
			return 1;
		else if (x < 0)
			return -1;
		return 0;

	}

	private boolean isSameClass(Instances data) {
		for (int i = 1; i < data.numInstances(); i++)
			if (data.instance(0).classValue() != data.instance(1).classValue())
				return false;
		return true;
	}

	public double lap(double sensitivity, double epsilon) {// epsilon-差分隐私
		double u = Math.random() - 0.5;
		return (sensitivity / epsilon) * sgn(u) * Math.log10(1.0 - 2.0 * Math.abs(u));
	}

	private double Count(Instances data, int[] attr, String[] attrValue) {
		double num = 0.0;
		for (int i = 0; i < data.numInstances(); i++) {
			boolean isEqual = true;
			for (int j = 0; j < attr.length; j++) {
				if (!data.instance(i).stringValue(attr[j]).equals(attrValue[j])) {
					isEqual = false;
					break;
				}
			}
			if (isEqual)
				num++;
		}
		return num;
	}

	/** 根据权重的比例随机选择一个索引 */
	private int selectByWeight(double[] weight) {
		double[] w = new double[weight.length];
		double sum = 0.0;
		for (int i = 0; i < weight.length; i++)
			sum += weight[i];
		// 归一化
		for (int i = 0; i < w.length; i++)
			w[i] = weight[i] / sum;

		double randomValue = new Random().nextDouble();
		for (int i = 0; i < w.length; i++) {
			if (w[i] == 0)
				continue;
			if (randomValue < w[i])
				return i;
			else
				randomValue -= w[i];
		}
		return w.length - 1;

	}

	/** 基尼指数的打分函数 */
	private double qGini(Instances data, int attr) {

		double qgini = 0.0;
		for (int i = 0; i < data.attribute(attr).numValues(); i++) {
			int[] attrs1 = new int[1];
			String[] attrValue1 = new String[1];
			attrs1[0] = attr;
			attrValue1[0] = data.attribute(attr).value(i);
			double TAJ = Count(data, attrs1, attrValue1);
			double t = 0.0;
			for (int j = 0; j < data.numClasses(); j++) {
				int[] attrs2 = new int[2];
				String[] attrValue2 = new String[2];
				attrs2[0] = attr;
				attrValue2[0] = data.attribute(attr).value(i);
				attrs2[1] = data.classIndex();
				attrValue2[1] = data.classAttribute().value(j);
				double TAJC = Count(data, attrs2, attrValue2);

				t += (TAJC / TAJ) * (TAJC / TAJ);
			}
			if (TAJ != 0)
				qgini -= (1 - t) * TAJ;
		}
		return qgini;

	}

	private double qMaxOperator(Instances data, int attr) {// Max Operator 的打分函数
		double qMax = 0.0;
		String[] attrValue = new String[2];
		int[] attrs = new int[2];
		attrs[0] = attr;
		attrs[1] = data.classIndex();
		for (int i = 0; i < data.attribute(attr).numValues(); i++) {
			attrValue[0] = data.attribute(attr).value(i);
			attrValue[1] = data.classAttribute().value(0);
			double tMax = Count(data, attrs, attrValue);
			for (int j = 1; j < data.classAttribute().numValues(); j++) {
				attrValue[1] = data.classAttribute().value(j);
				if (tMax < Count(data, attrs, attrValue))
					tMax = Count(data, attrs, attrValue);
			}
			qMax += tMax;
		}
		return qMax;
	}

	/**
	 * 根据最大类频数选择一个分裂属性，返回分裂属性索引值
	 * 
	 * @param data
	 *            数据集
	 * @param useDP
	 *            true:使用指数机制选择最佳分裂属性.false:不使用差分隐私选择最佳分裂属性
	 */
	private int bestSplit_MaxOperator(Instances data, int[] attrs, boolean useDP, double epsilon) {

		if (data.numInstances() == 0) {
			return attrs[new Random().nextInt(attrs.length)];
		}
		if (useDP) {// 使用指数机制选择最佳分裂属性
			double[] weight = new double[data.numAttributes() - 1];// 索引代表属性!
			for (int i = 0; i < attrs.length; i++) {
				weight[attrs[i]] = Math.exp(epsilon * qMaxOperator(data, attrs[i]) * 0.5);
			}
			return selectByWeight(weight);
		} else {
			double[] weight = new double[data.numAttributes() - 1];// 索引代表属性!
			for (int i = 0; i < attrs.length; i++) {
				weight[attrs[i]] = qMaxOperator(data, attrs[i]);
			}
			return Utils.maxIndex(weight);
		}
	}

	/** 根据信息增益选择最佳分裂属性 */
	private int bestSplit_IG(Instances data, int[] attrs, boolean useDP, double epsilon) throws Exception {
		if (data.numInstances() == 0) {
			return attrs[new Random().nextInt(attrs.length)];
		}
		if (useDP) {
			double[] weight = new double[data.numAttributes() - 1];// 索引代表属性!
			double sensitivity = Utils.log2(data.numClasses());
			// 计算每个属性(i)的权重
			for (int i = 0; i < attrs.length; i++) {
				double ig = computeInfoGain(data, data.attribute(attrs[i]));
				weight[attrs[i]] = Math.exp(epsilon * ig / 2 / sensitivity);
			}
			return selectByWeight(weight);
		} else {
			double[] weight = new double[data.numAttributes() - 1];
			for (int i = 0; i < attrs.length; i++) {
				weight[attrs[i]] = Math.exp(computeInfoGain(data, data.attribute(attrs[i])));
			}
			return Utils.maxIndex(weight);
		}
	}

	/** 根据基尼指数选择最佳分裂属性 */
	private int bestSplit_Gini(Instances data, int[] attrs, boolean useDP, double epsilon) throws Exception {
		if (data.numInstances() == 0) {
			return attrs[new Random().nextInt(attrs.length)];
		}
		if (useDP) {
			double[] weight = new double[data.numAttributes() - 1];// 索引代表属性!
			double sensitivity = 2.0;
			// 计算每个属性(i)的权重
			for (int i = 0; i < attrs.length; i++) {
				double gini = qGini(data, attrs[i]);
				weight[attrs[i]] = Math.exp(epsilon * gini / 2 / sensitivity);
			}
			return selectByWeight(weight);
		} else {
			double[] weight = new double[data.numAttributes() - 1];// 索引代表属性!
			for (int i = 0; i < attrs.length; i++) {
				weight[attrs[i]] = Math.exp(qGini(data, attrs[i]));
			}
			return Utils.maxIndex(weight);
		}
	}

	private int[] selectRandomAttr(int[] attrs, int num) {
		if (num == 0)
			return null;
		Random random = new Random();
		boolean[] used = new boolean[attrs.length];
		int t = 0;
		int[] newAttr = new int[num];
		for (int i = 0; i < num; i++) {
			int index = random.nextInt(attrs.length);
			if (!used[index]) {
				newAttr[t++] = attrs[index];
				used[index] = true;
			} else
				i--;
		}
		return newAttr;
	}

	/**
	 * Method for building an Id3 tree.
	 *
	 * @param data
	 *            the training data
	 * @exception Exception
	 *                if decision tree can't be built successfully
	 */
	private void makeTree(Instances data, int[] attrs, char qOption, int d) throws Exception {
		// 检测该结点是否要成为叶子结点
		if (attrs.length == 0 || d == 0) {// 成为叶子结点
			// if (data.numInstances() == 0) {
			// m_Attribute = null;
			// m_ClassValue = new Random().nextInt(data.classAttribute().numValues());
			// m_ClassAttribute = data.classAttribute();
			// return;
			// }
			Instances[] splitdata = splitData(data, data.classAttribute());
			double maxIndex = 0, maxNum = splitdata[0].numInstances();
			for (int i = 1; i < splitdata.length; i++) {
				double t = splitdata[i].numInstances();
				if (maxNum < t) {
					maxNum = t;
					maxIndex = i;
				}
			}
			m_Attribute = null;
			m_ClassValue = maxIndex;
			m_ClassAttribute = data.classAttribute();
			return;
		}

		// 不作为叶子结点，选择最佳分裂属性
		// 选择随机属性的个树
		int num = (int) Utils.log2(attrs.length);
		if (num == 0)
			num = 1;
		// 使用指数机制选择最佳分裂属性

		int bestSplitAttr = 1;
		if (qOption == 'M')
			bestSplitAttr = bestSplit_MaxOperator(data, selectRandomAttr(attrs, num), false, Double.NaN);
		else if (qOption == 'I')
			bestSplitAttr = bestSplit_IG(data, selectRandomAttr(attrs, num), false, Double.NaN);
		else if (qOption == 'G')
			bestSplitAttr = bestSplit_Gini(data, selectRandomAttr(attrs, num), false, Double.NaN);

		m_Attribute = data.attribute(bestSplitAttr);
		m_ClassAttribute = data.classAttribute();

		Instances[] splitdata = splitData(data, data.attribute(bestSplitAttr));
		m_Successors = new DiffPID3[m_Attribute.numValues()];
		m_numeric = false;
		if (originNumeric(data, bestSplitAttr)) {
			m_numeric = true;
			m_cutPoints = new double[m_Attribute.numValues() - 1];
			getCutPoints(data, bestSplitAttr);
		}
		// 深拷贝 去除已经用过的属性
		int[] t_attrs = new int[attrs.length - 1];
		int t = 0;
		for (int i = 0; i < attrs.length; i++) {
			if (t == t_attrs.length && bestSplitAttr != attrs[i])
				System.out.println(String.format("ERROR: %d", bestSplitAttr));
			if (bestSplitAttr != attrs[i])
				t_attrs[t++] = attrs[i];
		}
		// 递归构造
		for (int i = 0; i < splitdata.length; i++) {
			m_Successors[i] = new DiffPID3();
			if (splitdata[i].numInstances() == 0) {
				m_Successors[i].m_Attribute = null;
				m_Successors[i].m_ClassAttribute = data.classAttribute();
				m_Successors[i].m_ClassValue = Utils.maxIndex(classDistribution(data));
			} else
				m_Successors[i].makeTree(splitdata[i], t_attrs, qOption, d - 1);
		}
	}

	private void DiffPID3makeTree(Instances data, int[] attrs, char qOption, int d, double epsilon) throws Exception {
		// 检测该结点是否要成为叶子结点
		if (attrs.length == 0 || d == 0) {// 成为叶子结点
			Instances[] splitdata = splitData(data, data.classAttribute());
			double maxIndex = 0, maxNum = splitdata[0].numInstances() + lap(1, epsilon);// 第一次用到epsilon
			for (int i = 1; i < splitdata.length; i++) {
				double t = splitdata[i].numInstances() + lap(1, epsilon);
				if (maxNum < t) {
					maxNum = t;
					maxIndex = i;
				}
			}
			m_Attribute = null;
			m_ClassValue = maxIndex;
			m_ClassAttribute = data.classAttribute();
			return;
		}

		// 不作为叶子结点，选择最佳分裂属性
		// 选择随机属性的个树
		int num = (int) Utils.log2(attrs.length);
		if (num == 0)
			num = 1;
		// 使用指数机制选择最佳分裂属性

		int bestSplitAttr = 0;
		if (qOption == 'M')
			bestSplitAttr = bestSplit_MaxOperator(data, selectRandomAttr(attrs, num), true, epsilon);
		else if (qOption == 'I')
			bestSplitAttr = bestSplit_IG(data, selectRandomAttr(attrs, num), true, epsilon);
		else if (qOption == 'G')
			bestSplitAttr = bestSplit_Gini(data, selectRandomAttr(attrs, num), true, epsilon);

		m_Attribute = data.attribute(bestSplitAttr);
		m_ClassAttribute = data.classAttribute();

		Instances[] splitdata = splitData(data, data.attribute(bestSplitAttr));
		m_Successors = new DiffPID3[m_Attribute.numValues()];
		m_numeric = false;
		if (originNumeric(data, bestSplitAttr)) {
			m_numeric = true;
			m_cutPoints = new double[m_Attribute.numValues() - 1];
			getCutPoints(data, bestSplitAttr);
		}
		// 深拷贝 去除已经用过的属性
		int[] t_attrs = new int[attrs.length - 1];
		int t = 0;
		for (int i = 0; i < attrs.length; i++) {
			if (t == t_attrs.length && bestSplitAttr != attrs[i])
				System.out.println(bestSplitAttr);
			if (bestSplitAttr != attrs[i])
				t_attrs[t++] = attrs[i];
		}
		// 递归构造
		for (int i = 0; i < splitdata.length; i++) {
			m_Successors[i] = new DiffPID3();
			m_Successors[i].DiffPID3makeTree(splitdata[i], t_attrs, qOption, d - 1, epsilon);
		}
	}

	/**
	 * Builds Id3 decision tree classifier.
	 *
	 * @param data
	 *            the training data
	 * @exception Exception
	 *                if classifier can't be built successfully
	 */
	public void buildClassifier(Instances data, int[] attr, char qOption, int d) throws Exception {

		if (!data.classAttribute().isNominal()) {
			throw new UnsupportedClassTypeException("Id3: nominal class, please.");
		}
		Enumeration enumAtt = data.enumerateAttributes();
		while (enumAtt.hasMoreElements()) {
			if (!((Attribute) enumAtt.nextElement()).isNominal()) {
				throw new UnsupportedAttributeTypeException("Id3: only nominal " + "attributes, please.");
			}
		}
		Enumeration enu = data.enumerateInstances();
		while (enu.hasMoreElements()) {
			if (((Instance) enu.nextElement()).hasMissingValue()) {
				throw new NoSupportForMissingValuesException("Id3: no missing values, " + "please.");
			}
		}
		data = new Instances(data);
		data.deleteWithMissingClass();
		if (attr == null) {
			int[] attrs = new int[data.numAttributes() - 1];
			for (int i = 0; i < data.numAttributes() - 1; i++)
				attrs[i] = i;
			makeTree(data, attrs, qOption, d);
		} else {
			makeTree(data, attr, qOption, d);
		}
	}

	public void DiffPID3BuildClassifier(Instances data, int[] attr, char qOption, int d, double epsilon)
			throws Exception {

		if (!data.classAttribute().isNominal()) {
			throw new UnsupportedClassTypeException("Id3: nominal class, please.");
		}
		Enumeration enumAtt = data.enumerateAttributes();
		while (enumAtt.hasMoreElements()) {
			if (!((Attribute) enumAtt.nextElement()).isNominal()) {
				throw new UnsupportedAttributeTypeException("Id3: only nominal " + "attributes, please.");
			}
		}
		Enumeration enu = data.enumerateInstances();
		while (enu.hasMoreElements()) {
			if (((Instance) enu.nextElement()).hasMissingValue()) {
				throw new NoSupportForMissingValuesException("Id3: no missing values, " + "please.");
			}
		}
		data = new Instances(data);
		data.deleteWithMissingClass();

		if (attr == null) {
			int[] attrs = new int[data.numAttributes() - 1];
			for (int i = 0; i < data.numAttributes() - 1; i++)
				attrs[i] = i;
			DiffPID3makeTree(data, attrs, qOption, d, epsilon / (d + 1));
		} else {
			DiffPID3makeTree(data, attr, qOption, d, epsilon / (d + 1));
		}
	}

}
