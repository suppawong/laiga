import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.lazy.IBk;
import weka.classifiers.misc.InputMappedClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SMOTE;

public class Exp1 {
	
	public static HashMap<Instance, String> prepareData(Instances testData, Instances trainData, int foldIndex, String clStr) throws IOException {
		trainData.deleteAttributeAt(1467);
		trainData.deleteAttributeAt(1462);
		trainData.deleteAttributeAt(0);
		testData.deleteAttributeAt(1467);
		testData.deleteAttributeAt(1462);
		HashMap<Instance, String> sidMap = new HashMap<Instance, String>();
		if(clStr.equals("to")) {
			trainData.deleteAttributeAt(1463);
			trainData.deleteAttributeAt(1462);
			trainData.deleteAttributeAt(1461);
			testData.deleteAttributeAt(1464);
			testData.deleteAttributeAt(1463);
			testData.deleteAttributeAt(1462);
		}
		else if(clStr.equals("ad")) {
			trainData.deleteAttributeAt(1464);
			trainData.deleteAttributeAt(1463);
			trainData.deleteAttributeAt(1462);
			testData.deleteAttributeAt(1465);
			testData.deleteAttributeAt(1464);
			testData.deleteAttributeAt(1463);
		}
		else if(clStr.equals("in")) {
			trainData.deleteAttributeAt(1464);
			trainData.deleteAttributeAt(1463);
			trainData.deleteAttributeAt(1461);
			testData.deleteAttributeAt(1465);
			testData.deleteAttributeAt(1464);
			testData.deleteAttributeAt(1462);
		}
		else if(clStr.equals("co")) {
			trainData.deleteAttributeAt(1464);
			trainData.deleteAttributeAt(1462);
			trainData.deleteAttributeAt(1461);
			testData.deleteAttributeAt(1465);
			testData.deleteAttributeAt(1463);
			testData.deleteAttributeAt(1462);
		}
		else {
			return null;
		}
		
		for(int i = 0; i < testData.size(); i++) {
			sidMap.put(testData.get(i), String.valueOf((int)testData.get(i).value(0)));
			//System.out.println(String.valueOf((int)testTemp.get(i).value(0)));
		}
		testData.deleteAttributeAt(0);
		
		FileWriter fw = new FileWriter("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\test_new_" + clStr + ".arff");
		fw.write(testData.toString());
		fw.close();
		fw = new FileWriter("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\train_new_" + clStr + ".arff");
		fw.write(trainData.toString());
		fw.close();
		fw = new FileWriter("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\sid_" + clStr + ".csv");
		String backupID = "";
		for(int i = 0; i < testData.numInstances(); i++) {
			backupID += sidMap.get(testData.instance(i)) + "," + testData.instance(i) + "\n";
			//System.out.println(sidMap.get(testData.instance(i)) + "," + testData.instance(i));
		}
		fw.write(backupID);
		fw.close();
		return sidMap;
	}
	
	public static int numAttr;
	
	public static void runExp1a(int foldIndex, String cs, String fs) {
		try {
			//load data
			DataSource testSource = new DataSource("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\test.arff");
			DataSource trainSource = new DataSource("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\train.arff");
			Instances testData = testSource.getDataSet();
			Instances trainData = trainSource.getDataSet();
			String[] clStr = {"to", "ad", "in", "co"};
			for(String s : clStr) {
				Instances newTestData = new Instances(testData);
				Instances newTrainData = new Instances(trainData);
				HashMap<Instance, String> sidMap = prepareData(newTestData, newTrainData, foldIndex, s);
				int ci = newTestData.numAttributes() - 1;
				newTrainData.setClassIndex(ci);
				newTestData.setClassIndex(ci);
				Classifier classifier;
				InputMappedClassifier map = new InputMappedClassifier();
				weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
				weka.attributeSelection.ASEvaluation eval;
				weka.attributeSelection.ASSearch search;
				
				//run model
				switch(fs) {
				case "CE" :
					eval = new CfsSubsetEval();
					search = new BestFirst();
					filter.setEvaluator((CfsSubsetEval)eval);
					filter.setSearch((BestFirst)search);
					filter.setInputFormat(newTrainData);
					newTrainData = Filter.useFilter(newTrainData, filter);
					numAttr = newTrainData.numAttributes();
					break;
				default :
					//none
				}
				
				switch(cs) {
				case "LR" :
					classifier = new LinearRegression();
					map.setClassifier((LinearRegression)classifier);
					map.buildClassifier(newTrainData);
					break;
				case "SMOreg" :
					classifier = new SMOreg();
					map.setClassifier((SMOreg)classifier);
					map.buildClassifier(newTrainData);
					break;
				case "MLP" :
					classifier = new MultilayerPerceptron();
					map.setClassifier((MultilayerPerceptron)classifier);
					map.buildClassifier(newTrainData);
					break;
				case "IBk" :
					classifier = new IBk();
					map.setClassifier((IBk)classifier);
					map.buildClassifier(newTrainData);
					break;
				case "RF" :
					classifier = new RandomForest();
					map.setClassifier((RandomForest)classifier);
					map.buildClassifier(newTrainData);
					break;
				default :
					classifier = null;
					System.out.println("invalid classifier");
				}
				
				//output
				String addr = "D:\\Programming\\Java\\side project\\WekaAutomate\\exp1a\\fold" + foldIndex;
				String evalStr = "NumFeatures,Corr,MAPE,MAE,RMSE\n";
				KrirkWrapperEvaluation evalu = new KrirkWrapperEvaluation(newTrainData);
				evalu.evaluateModel(map, newTestData);
				System.out.println(evalu.toSummaryString());
				evalStr += newTrainData.numAttributes() + "," + evalu.correlationCoefficient() + "," +evalu.getMAPE() + ","
						+ evalu.meanAbsoluteError() + "," + evalu.rootMeanSquaredError();
				
				String predStr = "sid,actual,prediction\n";
				for(int i = 0; i < newTestData.numInstances(); i++) {
					Instance inst = newTestData.instance(i);
					double actualClass = inst.classValue();
					//System.out.println(actualClass);
					double predClass = map.classifyInstance(inst);
					//System.out.println(predClass);
					predStr += sidMap.get(inst) + "," + actualClass + "," + predClass + "\n";
					//System.out.println(sidMap.get(inst) + "," + actualClass + "," + predClass);
				}
				
				File f = new File(addr);
				if(!f.exists()) f.mkdir();
				//eval
				FileWriter fw = new FileWriter(addr + "\\" + cs + "." + fs + "." + s + ".eval.csv");
				fw.write(evalStr);
				fw.close();
				//pred
				fw = new FileWriter(addr + "\\" + cs + "." + fs + "." + s + ".pred.csv");
				fw.write(predStr);
				fw.close();
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}
	
	protected static void runMean(int foldIndex) throws Exception {
		DataSource testSource = new DataSource("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\test.arff");
		DataSource trainSource = new DataSource("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\train.arff");
		Instances testData = testSource.getDataSet();
		Instances trainData = trainSource.getDataSet();
		String[] clStr = {"to", "ad", "in", "co"};
		for(String s : clStr) {
			Instances newTestData = new Instances(testData);
			Instances newTrainData = new Instances(trainData);
			HashMap<Instance, String> sidMap = prepareData(newTestData, newTrainData, foldIndex, s);
			int ci = newTestData.numAttributes() - 1;
			newTrainData.setClassIndex(ci);
			newTestData.setClassIndex(ci);
		}
		
	}
	
	public static void main(String[] args) throws Exception {
		String arffFile = "G:\\Shared drives\\2021 MU Online Learning Research\\leadership-analysis\\for weka\\output_p.arff";
		DataSource source = new DataSource(arffFile);
		Instances data = source.getDataSet();
		int numfolds = 10;
		Main.splitData(10, data);
		String[] regressors = {"LR", "SMOreg", "MLP", "IBk", "RF"};
		String[] featureSelections = {"0", "CE"};
		
		//run Exp 1a
		for(int i = 0; i < numfolds; i++) {
			for(String r : regressors) {
				for(String f : featureSelections) {
					runExp1a(i, r, f);
				}
			}
		}
		
		//run Mean evaluation
		for(int i = 0; i < numfolds; i++) {
			runMean(i);
		}

	}

}
