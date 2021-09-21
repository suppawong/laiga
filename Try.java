import java.io.IOException;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;

public class Try {

	public static void main(String[] args) throws Exception {
		String arffFile = "G:\\Shared drives\\2021 MU Online Learning Research\\leadership-analysis\\for weka\\output_r.arff";
		DataSource source = new DataSource(arffFile);
		Instances data = source.getDataSet();
		System.out.println(data.attribute(0).name());
		/*Instances data2 = source.getDataSet();
		
		System.out.println("old: " + data.numAttributes());
		
		int numAttr = 0;
		int ci = data.numAttributes()-1;
		data.setClassIndex(ci);
		Classifier c;
		weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
		weka.attributeSelection.ASEvaluation eval;
		weka.attributeSelection.ASSearch search;

		eval = new CfsSubsetEval();
		search = new BestFirst();
		filter.setEvaluator((CfsSubsetEval)eval);
		filter.setSearch((BestFirst)search);
		filter.setInputFormat(data);
		data = Filter.useFilter(data, filter);
		
		System.out.println("new: " + data.numAttributes());
		numAttr = data.numAttributes();
		System.out.println("old: " + data2.numAttributes());
		eval = new InfoGainAttributeEval();
		search = new Ranker();
		((Ranker)search).setNumToSelect(numAttr-1);
		filter.setEvaluator((InfoGainAttributeEval)eval);
		filter.setSearch((Ranker)search);
		filter.setInputFormat(data2);
		data2 = Filter.useFilter(data2, filter);
		System.out.println("new: " + data2.numAttributes());
		
		LibLINEAR lib = new LibLINEAR();
		lib.buildClassifier(data);*/
	}

}
