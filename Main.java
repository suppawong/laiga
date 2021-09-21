
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.ClassBalancer;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.OptionHandler;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.misc.InputMappedClassifier;
import weka.classifiers.trees.RandomForest;


public class Main {
	
	public static void splitData(int num_folds, Instances data) throws IOException {
		Instances[] dataArr = new Instances[num_folds];
		for(int i = 0; i < num_folds; i++) {
			dataArr[i] = new Instances(data);
			dataArr[i].delete();
		}
		int index_0 = 0;
		int index_1 = 0;
		//System.out.println(dataSource.size());
		for(int i = 0; i < data.size(); i++) {
			Instance inst = new DenseInstance(data.get(i));
			if(inst.value(1467) == 0) {
				dataArr[index_0].add(inst);
				//System.out.println("0: " + i + " " + (int)inst.value(0));
				index_0++;
				if(index_0 == num_folds)
					index_0 = 0;
			}
			else if(inst.value(1467) == 1) {
				dataArr[index_1].add(inst);
				//System.out.println("1: " + i + " " + (int)inst.value(0));
				index_1++;
				if(index_1 == num_folds)
					index_1 = 0;
			}
		}
		FileWriter fw;
		File f;
		for(int i = 0; i < num_folds; i++) {
			f = new File("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + i);
			if(!f.exists())
				f.mkdir();
			Instances testData = new Instances(dataArr[i]);
			fw = new FileWriter("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + i + "\\test.arff");
			fw.write(testData.toString());
			fw.close();
			Instances trainData = new Instances(dataArr[i]);
			trainData.delete();
			for(int j = 0; j < num_folds; j++) {
				if(j != i) {
					trainData.addAll(dataArr[j]);
				}
			}
			fw = new FileWriter("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + i + "\\train.arff");
			fw.write(trainData.toString());
			fw.close();
		}
		
	}
	
	/*public static Evaluation run(int foldIndex, DataSource testSource, DataSource trainSource) {
		try {
			Instances testData = testSource.getDataSet();
			Instances trainData = trainSource.getDataSet();
			FilteredClassifier c1 = new FilteredClassifier();
			Remove f1 = new Remove();
			f1.setAttributeIndices("1");
			FilteredClassifier c2 = new FilteredClassifier();
			AttributeSelection f2 = new AttributeSelection();
			CfsSubsetEval eval = new CfsSubsetEval();
			BestFirst search = new BestFirst();
			search.setOptions(weka.core.Utils.splitOptions("-D 1 -N 5"));
			f2.setEvaluator(eval);
			f2.setSearch(search);
			ThresholdSelector c3 = new ThresholdSelector();
			RandomForest c4 = new RandomForest();
			c4.setNumExecutionSlots(8);
			c4.setNumIterations(300);
			c3.setClassifier(c4);
			c2.setClassifier(c3);
			c2.setFilter(f2);
			c1.setClassifier(c2);
			c1.setFilter(f1);
			trainData.deleteAttributeAt(0);
			HashMap<Instance, String> sidMap = new HashMap<Instance, String>();
			for(int i = 0; i < testData.size(); i++) {
				sidMap.put(testData.get(i), String.valueOf((int)testData.get(i).value(0)));
			}
			testData.deleteAttributeAt(0);
			FileWriter fw = new FileWriter("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\test_new.arff");
			fw.write(testData.toString());
			fw.close();
			fw = new FileWriter("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\train_new.arff");
			fw.write(trainData.toString());
			fw.close();
			fw = new FileWriter("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\sid.csv");
			String backupID = "";
			for(int i = 0; i < testData.numInstances(); i++) {
				backupID += sidMap.get(testData.instance(i)) + "," + testData.instance(i) + "\n";
			}
			fw.write(backupID);
			fw.close();
			int ci = testData.numAttributes()-1;
			trainData.setClassIndex(ci);
			testData.setClassIndex(ci);
			Classifier c;
			c = new RandomForest();
			((RandomForest)c).setOptions(Utils.splitOptions("-I 100 -K 0 -S 1 -num-slots 8"));
			((RandomForest)c).buildClassifier(trainData);
			String predStr = "sid,actual,prediction\r\n";
			for(int i = 0; i < testData.numInstances(); i++) {
				double actualClass = testData.instance(i).classValue();
				String actual = testData.classAttribute().value((int)actualClass);
				Instance inst = testData.instance(i);
				double predRF = ((RandomForest)c).classifyInstance(inst);
				String pred = testData.classAttribute().value((int)predRF);
				predStr += sidMap.get(inst) + "," + actual + "," + pred + "\n";
			}
			fw = new FileWriter("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\pred.csv");
			fw.write(predStr);
			fw.close();
			Evaluation eval = new Evaluation(trainData);
			eval.evaluateModel(c, testData);
			String evalStr = "NumFeatures,Precision,Recall,F1,MCC,ROC,PRC\n";
			evalStr += (ci+1) + "," + eval.precision(1) + "," + eval.recall(1) + "," + eval.fMeasure(1) + ","
					+ eval.matthewsCorrelationCoefficient(1) + "," + eval.areaUnderROC(1) + ","
					+ eval.areaUnderPRC(1);
			fw = new FileWriter("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\eval.csv");
			fw.write(evalStr);
			fw.close();
			return eval;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}*/
	
	public static HashMap<Instance, String> prepareData(Instances testData, Instances trainData, int foldIndex) throws IOException {
		trainData.deleteAttributeAt(0);
		HashMap<Instance, String> sidMap = new HashMap<Instance, String>();
		for(int i = 0; i < testData.size(); i++) {
			sidMap.put(testData.get(i), String.valueOf((int)testData.get(i).value(0)));
		}
		testData.deleteAttributeAt(0);
		FileWriter fw = new FileWriter("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\test_new.arff");
		fw.write(testData.toString());
		fw.close();
		fw = new FileWriter("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\train_new.arff");
		fw.write(trainData.toString());
		fw.close();
		fw = new FileWriter("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\sid.csv");
		String backupID = "";
		for(int i = 0; i < testData.numInstances(); i++) {
			backupID += sidMap.get(testData.instance(i)) + "," + testData.instance(i) + "\n";
		}
		fw.write(backupID);
		fw.close();
		return sidMap;
	}
	
	public static int numAttr;
	
	public static void runExp2a(int foldIndex, String cs, String ds, String fs) {
		try {
			//load data
			DataSource testSource = new DataSource("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\test.arff");
			DataSource trainSource = new DataSource("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + foldIndex + "\\train.arff");
			Instances testData = testSource.getDataSet();
			Instances trainData = trainSource.getDataSet();
			HashMap<Instance, String> sidMap = prepareData(testData, trainData, foldIndex);
			
			int ci = testData.numAttributes()-1;
			trainData.setClassIndex(ci);
			testData.setClassIndex(ci);
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
				filter.setInputFormat(trainData);
				trainData = Filter.useFilter(trainData, filter);
				numAttr = trainData.numAttributes();
				
				break;
			case "IG" :
				eval = new InfoGainAttributeEval();
				search = new Ranker();
				((Ranker)search).setNumToSelect(numAttr-1);
				filter.setEvaluator((InfoGainAttributeEval)eval);
				filter.setSearch((Ranker)search);
				filter.setInputFormat(trainData);
				trainData = Filter.useFilter(trainData, filter);
				break;
			default :
				//none
			}
			
			
			weka.filters.Filter balancer;
			switch(ds) {
			case "RS" :
				balancer = new Resample();
				((Resample)balancer).setSampleSizePercent(118.75);
				((Resample)balancer).setInputFormat(trainData);
				trainData = Filter.useFilter(trainData, balancer);
				break;
			case "SMOTE" :
				balancer = new SMOTE();
				((SMOTE)balancer).setInputFormat(trainData);
				trainData = Filter.useFilter(trainData, balancer);
				break;
			default :
				//none
			}
			
			switch(cs) {
			case "NB" :
				classifier = new NaiveBayes();
				map.setClassifier((NaiveBayes)classifier);
				map.buildClassifier(trainData);
				break;
			case "LR" :
				classifier = new Logistic();
				map.setClassifier((Logistic)classifier);
				map.buildClassifier(trainData);
				break;
			case "LL" :
				classifier = new LibLINEAR();
				((LibLINEAR) classifier).setOptions(Utils.splitOptions("-S 0 -C 1.0 -E 0.01 -B 1.0 -Z -P"));
				((LibLINEAR) classifier).setProbabilityEstimates(true);
				map.setClassifier((LibLINEAR)classifier);
				map.buildClassifier(trainData);
				break;
			/*case "SVM" :
				classifier = new weka.classifiers.functions.LibSVM();
				((weka.classifiers.functions.LibSVM) classifier).setOptions(Utils.splitOptions("-S 0 -K 0 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1 -B -Z "));
				break;*/
			case "IBk" :
				classifier = new IBk();
				map.setClassifier((IBk)classifier);
				map.buildClassifier(trainData);
				break;
			case "RF" :
				classifier = new RandomForest();
				map.setClassifier((RandomForest)classifier);
				map.buildClassifier(trainData);
				break;
			default :
				classifier = null;
				System.out.println("invalid classifier");
			}
			
			//output
			String addr = "D:\\Programming\\Java\\side project\\WekaAutomate\\exp2a\\fold" + foldIndex;
			String evalStr = "NumFeatures,Precision,Recall,F1,MCC,ROC,PRC\n";
			Evaluation evalu = new Evaluation(trainData);
			evalu.evaluateModel(map, testData);
			evalStr += trainData.numAttributes() + "," + evalu.precision(1) + "," + evalu.recall(1) + "," + evalu.fMeasure(1) + ","
					+ evalu.matthewsCorrelationCoefficient(1) + "," + evalu.areaUnderROC(1) + ","
					+ evalu.areaUnderPRC(1);
			
			String predStr = "sid,actual,prediction\n";
			for(int i = 0; i < testData.numInstances(); i++) {
				double actualClass = testData.instance(i).classValue();
				String actual = testData.classAttribute().value((int)actualClass);
				Instance inst = testData.instance(i);
				double predClass = map.classifyInstance(inst);
				String pred = testData.classAttribute().value((int)predClass);
				predStr += sidMap.get(inst) + "," + actual + "," + pred + "\n";
			}
			
			File f = new File(addr);
			if(!f.exists()) f.mkdir();
			//eval
			FileWriter fw = new FileWriter(addr + "\\" + cs + "." + ds + "." + fs + ".eval.csv");
			fw.write(evalStr);
			fw.close();
			//pred
			fw = new FileWriter(addr + "\\" + cs + "." + ds + "." + fs + ".pred.csv");
			fw.write(predStr);
			fw.close();
			
			
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}
	
	public static void main(String[] args) throws Exception {
		String arffFile = "G:\\Shared drives\\2021 MU Online Learning Research\\leadership-analysis\\for weka\\output_r.arff";
		DataSource source = new DataSource(arffFile);
		Instances data = source.getDataSet();
		int numfolds = 10;
		splitData(10, data);
		/*double prec = 0;
		double rec = 0;
		double f1 = 0;
		double mcc = 0;
		double roc = 0;
		double prc = 0;
		for(int i = 0; i < numfolds; i++) {
			DataSource testSource = new DataSource("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + i + "\\test.arff");
			DataSource trainSource = new DataSource("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\fold" + i + "\\train.arff");
			Evaluation eval = run(i, testSource, trainSource);
			prec += eval.precision(1);
			rec += eval.recall(1);
			f1 += eval.fMeasure(1);
			mcc += eval.matthewsCorrelationCoefficient(1);
			roc += eval.areaUnderROC(1);
			prc += eval.areaUnderPRC(1);
		}
		String evalStr = "NumFeatures,Precision,Recall,F1,MCC,ROC,PRC\n";
		FileWriter fw = new FileWriter("D:\\Programming\\Java\\side project\\WekaAutomate\\folds\\evalsum.csv");
		evalStr += (data.numAttributes() - 1) + "," + (prec / numfolds) + "," + (rec / numfolds) + "," + (f1 / numfolds) + "," 
		+ (mcc / numfolds) + "," + (roc / numfolds) + "," + (prc / numfolds);
		fw.write(evalStr);
		fw.close();*/
		String[] classifiers = {"NB", "LR", "LL", "IBk", "RF"};
		//String[] classifiers = {"LL"};
		String[] dataBalancings = {"0", "RS", "SMOTE"};
		String[] featureSelections = {"0", "CE", "IG"};
		
		//run Exp 2a
		for(int i = 0; i < numfolds; i++) {
			for(String c : classifiers) {
				for(String d : dataBalancings) {
					for(String f : featureSelections) {
						runExp2a(i, c, d, f);
					}
				}
			}
			
		}
		
	}

}
