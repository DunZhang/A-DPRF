package test;

import java.io.FileReader;
import java.util.Random;

import weka.core.Instances;

public class Test {

	public static void main(String[] args) throws Exception {
		// a simple example to use our algorithm
		// if you have any problems, please contact dunnzhang0@gmail.com
		int treeNum = 20, d = 5, N1 = 15;
		long seed = new Random().nextLong();
		double epsilon = 3;
		char qOption = 'I', Sampling = 'B', overSamplingOption = 'S';
		Instances data=new Instances(new FileReader("D:\\DataSets\\PROMISE\\ant-1.5.arff"));
		
		
		
		double testResult=TestTrees.DPRF_HoldOut(data, treeNum, d, epsilon, qOption, Sampling, overSamplingOption, N1, seed);
		System.out.println(testResult);
	}
}