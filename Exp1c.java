import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.supportVector.RegSMO;
import weka.classifiers.functions.supportVector.RegSMOImproved;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RemoveByName;

public class Exp1c {

	public static void main(String[] args) throws Exception {
		for(int i = 0; i < 11; i++) {
			String s = "Y" + ((i / 3) + 1) + "S" + ((i % 3) + 1);
			runTotal(s);
			runAdmin(s);
			runInter(s);
			runConcept(s);
		}
		System.out.println("done");
	}

	protected static void runTotal(String sem) throws Exception {
		String sourceStr = "G:\\Shared drives\\2021 MU Online Learning Research\\"
				+ "leadership-analysis\\for weka\\output_p_" + sem + ".arff";
		DataSource source = new DataSource(sourceStr);
		Instances data = source.getDataSet();
		int numfolds = 10;
		RemoveByName filter1 = new RemoveByName();
		filter1.setExpression("sid|participated|administrative|interpersonal|conceptual|mentioned");
		filter1.setInputFormat(data);
		data = Filter.useFilter(data, filter1);
		//System.out.println(data.toString());
		weka.filters.supervised.attribute.AttributeSelection filter2 = new weka.filters.supervised.attribute.AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();;
		BestFirst search = new BestFirst();
		filter2.setEvaluator((CfsSubsetEval)eval);
		filter2.setSearch((BestFirst)search);
		filter2.setInputFormat(data);
		data = Filter.useFilter(data, filter2);
		data.setClassIndex(data.numAttributes() - 1);
		SMOreg classifier = new SMOreg();
		RegSMO opt = new RegSMO();
		String options[] = {"-P 1.0E-12", "-L 0.001", "-W 1"};
		opt.setOptions(options);
		classifier.setRegOptimizer(opt);
		classifier.buildClassifier(data);
		KrirkWrapperEvaluation evaluation = new KrirkWrapperEvaluation(data);
		evaluation.crossValidateModel(classifier, data, numfolds, new Random(1));
		output(data, evaluation, "total", sem);
	}
	
	protected static void runAdmin(String sem) throws Exception {
		String sourceStr = "G:\\Shared drives\\2021 MU Online Learning Research\\"
				+ "leadership-analysis\\for weka\\output_p_" + sem + ".arff";
		DataSource source = new DataSource(sourceStr);
		Instances data = source.getDataSet();
		int numfolds = 10;
		RemoveByName filter1 = new RemoveByName();
		filter1.setExpression("sid|participated|total|interpersonal|conceptual|mentioned");
		filter1.setInputFormat(data);
		data = Filter.useFilter(data, filter1);
		//System.out.println(data.toString());
		weka.filters.supervised.attribute.AttributeSelection filter2 = new weka.filters.supervised.attribute.AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();;
		BestFirst search = new BestFirst();
		filter2.setEvaluator((CfsSubsetEval)eval);
		filter2.setSearch((BestFirst)search);
		filter2.setInputFormat(data);
		data = Filter.useFilter(data, filter2);
		data.setClassIndex(data.numAttributes() - 1);
		SMOreg classifier = new SMOreg();
		RegSMOImproved opt = new RegSMOImproved();
		String options[] = {"-T 0.081", "-V", "-P 1.0E-12", "-L 0.021", "-W 1"};
		opt.setOptions(options);
		classifier.setRegOptimizer(opt);
		classifier.buildClassifier(data);
		KrirkWrapperEvaluation evaluation = new KrirkWrapperEvaluation(data);
		evaluation.crossValidateModel(classifier, data, numfolds, new Random(1));
		output(data, evaluation, "admin", sem);
	}
	
	protected static void runInter(String sem) throws Exception {
		String sourceStr = "G:\\Shared drives\\2021 MU Online Learning Research\\"
				+ "leadership-analysis\\for weka\\output_p_" + sem + ".arff";
		DataSource source = new DataSource(sourceStr);
		Instances data = source.getDataSet();
		int numfolds = 10;
		RemoveByName filter1 = new RemoveByName();
		filter1.setExpression("sid|participated|administrative|total|conceptual|mentioned");
		filter1.setInputFormat(data);
		data = Filter.useFilter(data, filter1);
		//System.out.println(data.toString());
		weka.filters.supervised.attribute.AttributeSelection filter2 = new weka.filters.supervised.attribute.AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();;
		BestFirst search = new BestFirst();
		filter2.setEvaluator((CfsSubsetEval)eval);
		filter2.setSearch((BestFirst)search);
		filter2.setInputFormat(data);
		data = Filter.useFilter(data, filter2);
		data.setClassIndex(data.numAttributes() - 1);
		SMOreg classifier = new SMOreg();
		RegSMOImproved opt = new RegSMOImproved();
		String options[] = {"-T 0.0116", "-V", "-P 1.0E-12", "-L 0.05", "-W 1"};
		opt.setOptions(options);
		classifier.setRegOptimizer(opt);
		classifier.buildClassifier(data);
		KrirkWrapperEvaluation evaluation = new KrirkWrapperEvaluation(data);
		evaluation.crossValidateModel(classifier, data, numfolds, new Random(1));
		output(data, evaluation, "inter", sem);
	}
	
	protected static void runConcept(String sem) throws Exception {
		String sourceStr = "G:\\Shared drives\\2021 MU Online Learning Research\\"
				+ "leadership-analysis\\for weka\\output_p_" + sem + ".arff";
		DataSource source = new DataSource(sourceStr);
		Instances data = source.getDataSet();
		int numfolds = 10;
		RemoveByName filter1 = new RemoveByName();
		filter1.setExpression("sid|participated|administrative|interpersonal|total|mentioned");
		filter1.setInputFormat(data);
		data = Filter.useFilter(data, filter1);
		//System.out.println(data.toString());
		weka.filters.supervised.attribute.AttributeSelection filter2 = new weka.filters.supervised.attribute.AttributeSelection();
		CfsSubsetEval eval = new CfsSubsetEval();;
		BestFirst search = new BestFirst();
		filter2.setEvaluator((CfsSubsetEval)eval);
		filter2.setSearch((BestFirst)search);
		filter2.setInputFormat(data);
		data = Filter.useFilter(data, filter2);
		data.setClassIndex(data.numAttributes() - 1);
		SMOreg classifier = new SMOreg();
		RegSMOImproved opt = new RegSMOImproved();
		String options[] = {"-T 0.15", "-V", "-P 1.0E-12", "-L 0.1", "-W 1"};
		opt.setOptions(options);
		classifier.setRegOptimizer(opt);
		classifier.buildClassifier(data);
		KrirkWrapperEvaluation evaluation = new KrirkWrapperEvaluation(data);
		evaluation.crossValidateModel(classifier, data, numfolds, new Random(1));
		output(data, evaluation, "concept", sem);
	}
	
	protected static void output(Instances data, KrirkWrapperEvaluation eval, String cls, String sem) throws Exception {
		String addr = "D:\\Programming\\Java\\side project\\WekaAutomate\\exp1c\\";
		String evalStr = "NumFeatures,Corr,MAPE,MAE,RMSE\n";
		//System.out.println(eval.toSummaryString());
		evalStr += data.numAttributes() + "," + eval.correlationCoefficient() + "," +eval.getMAPE() + ","
				+ eval.meanAbsoluteError() + "," + eval.rootMeanSquaredError();
		FileWriter fw = new FileWriter(addr + cls + "." + sem + ".csv");
		fw.write(evalStr);
		fw.close();
	}

}
