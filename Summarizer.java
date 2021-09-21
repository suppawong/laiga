import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class Summarizer {

	protected static void summarizeExp2a(int numfolds, FileWriter fw) throws IOException {
		FileReader fr;
		BufferedReader br;
		String[] classifiers = {"NB", "LR", "LL", "IBk", "RF"};
		String[] dataBalancings = {"0", "RS", "SMOTE"};
		String[] featureSelections = {"0", "CE", "IG"};
		for(String c : classifiers) {
			for(String d : dataBalancings) {
				for(String f : featureSelections) {
					double[][] dataArr = new double[numfolds][7];
					for(int i = 0; i < numfolds; i++) {
						fr = new FileReader("D:\\Programming\\Java\\side project\\WekaAutomate\\exp2a\\fold"
								+ i + "\\" + c + "." + d + "." + f + ".eval.csv");
						br = new BufferedReader(fr);
						String cur_line = br.readLine();
			            //String[] headerArr = cur_line.split(",");
			            cur_line = br.readLine();
			            String[] dataStrArr = cur_line.split(",");
			            int j = 0;
			            for(String s : dataStrArr) {
			            	dataArr[i][j] = Double.parseDouble(s);
			            	j++;
			            }
					}
					double[] avgData = new double[7];
					for(int j = 0; j < 7; j++) {
						double sum = 0.0;
						for(int i = 0; i < numfolds; i++) {
							sum += dataArr[i][j];
						}
						avgData[j] = sum / (double)numfolds;
					}
					String[] comb = {c, f, d};
					summarizeToFileExp2a(avgData, comb, fw);
				}
			}
		}
		fw.close();
	}
	
	protected static void summarizeToFileExp2a(double[] avgData, String[] comb, FileWriter fw) {
		try {
			String toPrint = comb[0] + "," + comb[1] + "," + comb[2] + ",";
			for(double d : avgData) {
				toPrint += d + ",";
			}
			toPrint = toPrint.substring(0, toPrint.length()-1);
			toPrint += "\n";
			fw.write(toPrint);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
 	
	protected static void summarizeExp1a(int numfolds, FileWriter fw) throws IOException {
		FileReader fr;
		BufferedReader br;
		String[] regressors = {"LR", "SMOreg", "MLP", "IBk", "RF"};
		String[] featureSelections = {"0", "CE"};
		String[] clStr = {"to", "ad", "in", "co"};
		for(String c : regressors) {
			for(String f : featureSelections) {
				for(String s : clStr) {
					double[][] dataArr = new double[numfolds][5];
					for(int i = 0; i < numfolds; i++) {
						fr = new FileReader("D:\\Programming\\Java\\side project\\WekaAutomate\\exp1a\\fold"
								+ i + "\\" + c + "." + f + "." + s + ".eval.csv");
						br = new BufferedReader(fr);
						String cur_line = br.readLine();
						//String[] headerArr = cur_line.split(",");
						cur_line = br.readLine();
						String[] dataStrArr = cur_line.split(",");
						int j = 0;
						for(String d : dataStrArr) {
							dataArr[i][j] = Double.parseDouble(d);
							j++;
						}
					}
					double[] avgData = new double[5];
					for(int j = 0; j < 5; j++) {
						double sum = 0.0;
						for(int i = 0; i < numfolds; i++) {
							sum += dataArr[i][j];
						}
						avgData[j] = sum / (double)numfolds;
					}
					String[] comb = {c, f, s};
					summarizeToFileExp1a(avgData, comb, fw);
				}
			}
		}
		fw.close();
	}
	
	protected static void summarizeToFileExp1a(double[] avgData, String[] comb, FileWriter fw) {
		try {
			String toPrint = comb[0] + "," + comb[1] + "," + comb[2] + ",";
			for(double d : avgData) {
				toPrint += d + ",";
			}
			toPrint = toPrint.substring(0, toPrint.length()-1);
			toPrint += "\n";
			fw.write(toPrint);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		try {
			//file header
			/*FileWriter fw = new FileWriter("D:\\Programming\\Java\\side project\\WekaAutomate\\exp2a\\summary.csv");
			fw.write("Classifier,FeatureSelection,DataBalancing,NumFeatures,Precision,Recall,F1,MCC,ROC,PRC\n");
			summarizeExp2a(10, fw);*/
			
			FileWriter fw = new FileWriter("D:\\Programming\\Java\\side project\\WekaAutomate\\exp1a\\summary.csv");
			fw.write("Regressor,FeatureSelection,TargetAttr,NumFeatures,Corr,MAPE,MAE,RMSE\n");
			summarizeExp1a(10, fw);
			System.out.println("done");
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}

}
