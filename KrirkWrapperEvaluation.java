import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class KrirkWrapperEvaluation extends Evaluation {

	public KrirkWrapperEvaluation(Instances data) throws Exception {
		super(data);
		// TODO Auto-generated constructor stub
		m_delegate = new KrirkRealEvaluation(data);
	}
	
	public KrirkWrapperEvaluation(Instances data, CostMatrix costMatrix) throws Exception {
	    super(data,costMatrix);
	    m_delegate = new KrirkRealEvaluation(data);
	 }
	
	public double getMAPE()
	{
		return ((KrirkRealEvaluation)m_delegate).getMAPE();
	}

}
