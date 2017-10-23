package test;

import java.io.FileReader;
import java.util.Random;

import weka.core.Instances;

public class Test {

	public static void main(String[] args) throws Exception {
		//a simple example to use our algorithm
		// if you have any problems, please contact dunnzhang0@gmail.com
		String filePath = "D:\\DataSets\\PROMISE\\ant-1.5.arff";
		int treeNum = 20, d = 5 ,N=15;
		long seed = new Random().nextLong();
		double epsilon = 3;
		Instances data = new Instances(new FileReader(filePath));
		double result=TestTrees.DPRF_HoldOut(data, treeNum, d, epsilon, 'I', 'A', 'S', N, seed);
		System.out.println(result);
		
	}
}