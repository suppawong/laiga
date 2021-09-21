import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveByName;

public class AttributeFilter {

	public static void main(String[] args) throws Exception {
		String pFile = "output_p";
		String rFile = "output_r";
		String arffFile = "G:\\Shared drives\\2021 MU Online Learning Research\\leadership-analysis\\for weka\\" + pFile + ".arff";
		DataSource source = new DataSource(arffFile);
		Instances data = new Instances(source.getDataSet());
		
		//Instances outputData = new Instances(onlyGrade(data));
		//String outputAddr = "G:\\Shared drives\\2021 MU Online Learning Research\\leadership-analysis\\for weka\\" + rFile + "_grade.arff";
		
		for(int i = 0; i < 11; i++) {
			String s = "Y" + ((i / 3) + 1) + "S" + ((i % 3) + 1);
			String outputAddr = "G:\\Shared drives\\2021 MU Online Learning Research\\leadership-analysis\\for weka\\" + pFile + "_" + s + ".arff";
			Instances outputData = onlySem(data, i);
			ArffSaver as = new ArffSaver();
			as.setInstances(outputData);
			as.setFile(new File(outputAddr));
			as.writeBatch();
			
			//System.out.println(s);
		}
		System.out.println("done");
		/*String outputAddr = "G:\\Shared drives\\2021 MU Online Learning Research\\leadership-analysis\\for weka\\" + rFile + "_.arff";
		ArffSaver as = new ArffSaver();
		as.setInstances(outputData);
		as.setFile(new File(outputAddr));
		as.writeBatch();*/
	}
	
	public static Instances onlyGrade(Instances data) {
		int size = data.numAttributes();
		for(int i = 0; i < size; i++) {
			if(data.attribute(i).name().charAt(0) == 'm') {
				System.out.println(data.attribute(i).name());
				data.deleteAttributeAt(i);
			}
		}
		return data;
	}
	
	public static Instances onlySem(Instances data, int sem) throws Exception {
		Instances tempData = new Instances(data);
		for(int i = 10; i >= 0; i--) {
			if(sem == i) {
				break;
			}
			String s = "Y" + ((i / 3) + 1) + "S" + ((i % 3) + 1);
			RemoveByName filter = new RemoveByName();
			filter.setExpression(".*" + s + ".*");
			filter.setInputFormat(tempData);
			tempData = Filter.useFilter(tempData, filter);
		}
		return tempData;
	}
}
