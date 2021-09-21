import weka.classifiers.CostMatrix;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public class KrirkRealEvaluation extends Evaluation {
	//Added the ability to compute MAPE
	protected double m_SumAbsPctErr;
	
	
	public KrirkRealEvaluation(Instances data) throws Exception {
		super(data);
	}
	
	 public KrirkRealEvaluation(Instances data, CostMatrix costMatrix) throws Exception {
		 super(data, costMatrix);
	 }

	public final double getMAPE() {

	    return m_SumAbsPctErr / (m_WithClass - m_Unclassified);
	  }
	
	@Override
	protected void updateNumericScores(double[] predicted, double[] actual,
		    double weight)
	{
		double diff;
	    double sumErr = 0, sumAbsErr = 0, sumSqrErr = 0;
	    //krirk
	    double sumAbsPctErr = 0;
	    double sumPriorAbsErr = 0, sumPriorSqrErr = 0;
	    for (int i = 0; i < m_NumClasses; i++) {
	      diff = predicted[i] - actual[i];
	      sumErr += diff;
	      sumAbsErr += Math.abs(diff);
	      //Krirk
	      sumAbsPctErr += Math.abs(diff/actual[i]);
	      sumSqrErr += diff * diff;
	      diff = (m_ClassPriors[i] / m_ClassPriorsSum) - actual[i];
	      sumPriorAbsErr += Math.abs(diff);
	      sumPriorSqrErr += diff * diff;
	    }
	    m_SumErr += weight * sumErr / m_NumClasses;
	    m_SumAbsErr += weight * sumAbsErr / m_NumClasses;
	    //Krirk
	    m_SumAbsPctErr += weight*sumAbsPctErr/m_NumClasses;
	    		
	    m_SumSqrErr += weight * sumSqrErr / m_NumClasses;
	    m_SumPriorAbsErr += weight * sumPriorAbsErr / m_NumClasses;
	    m_SumPriorSqrErr += weight * sumPriorSqrErr / m_NumClasses;
		
	}
	
}
