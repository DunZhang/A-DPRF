package utils;

import weka.core.Utils;

public abstract class StatisticsUtils {
	/** 输出均值和标准差 */
	public static double[] StandardDeviation(double[] a) throws Exception {
		double sum = Utils.sum(a);
		double avg = sum / a.length;
		double d = 0;
		double[] res = new double[2];
		for (int i = 0; i < a.length; i++)
			d += (avg - a[i]) * (avg - a[i]);
		d /= a.length;
		res[0] = avg;
		res[1] = Math.sqrt(d);
		return res;
	}

}
