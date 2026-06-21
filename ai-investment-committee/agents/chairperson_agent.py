class ChairpersonAgent:
    @staticmethod
    def aggregate(agent_outputs):
        """
        Aggregate recommendations from various agents and provide a final recommendation.
        
        Parameters:
        - agent_outputs: List of dictionaries containing agent recommendations and confidence levels.
        
        Returns:
        - A dictionary containing the final recommendation and confidence level.
        """
        recommendations = {}
        
        for output in agent_outputs:
            recommendation = output.get("result", {}).get("recommendation")
            confidence = output.get("result", {}).get("confidence", 0)
            recommendations[recommendation] = recommendations.get(recommendation, 0) + confidence
        
        final_recommendation = max(recommendations, key=recommendations.get, default="HOLD")
        final_confidence = recommendations.get(final_recommendation, 0) // len(agent_outputs)
        
        return {
            "final_recommendation": final_recommendation,
            "final_confidence": final_confidence,
            "votes": recommendations
        }