
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sqlite3
import json

class RiskAnalyzer:
    def __init__(self, db_file="satta_analytics.db"):
        self.db_file = db_file
        self.scaler = StandardScaler()
        
    def calculate_market_volatility(self, df, market, window=30):
        """Calculate market volatility over specified window"""
        market_data = df[df["Market"] == market].tail(window)
        
        if len(market_data) < 10:
            return {"volatility": 0, "risk_level": "UNKNOWN"}
        
        # Calculate price volatility
        open_returns = market_data["Open"].pct_change().dropna()
        close_returns = market_data["Close"].pct_change().dropna()
        
        open_volatility = open_returns.std() * np.sqrt(window)
        close_volatility = close_returns.std() * np.sqrt(window)
        
        avg_volatility = (open_volatility + close_volatility) / 2
        
        # Risk levels based on volatility
        if avg_volatility < 0.1:
            risk_level = "LOW"
        elif avg_volatility < 0.2:
            risk_level = "MEDIUM"
        elif avg_volatility < 0.3:
            risk_level = "HIGH"
        else:
            risk_level = "EXTREME"
        
        return {
            "volatility": avg_volatility,
            "open_volatility": open_volatility,
            "close_volatility": close_volatility,
            "risk_level": risk_level
        }
    
    def analyze_pattern_strength(self, df, market, pattern_window=50):
        """Analyze strength of number patterns"""
        market_data = df[df["Market"] == market].tail(pattern_window)
        
        if len(market_data) < 20:
            return {"strength": 0, "patterns": []}
        
        # Analyze number frequency patterns
        all_numbers = list(market_data["Open"]) + list(market_data["Close"])
        number_freq = pd.Series(all_numbers).value_counts()
        
        # Calculate pattern strength based on distribution
        max_freq = number_freq.max()
        min_freq = number_freq.min()
        freq_range = max_freq - min_freq
        
        # Normalized pattern strength (0-1)
        pattern_strength = min(freq_range / len(market_data), 1.0)
        
        # Identify strong patterns
        threshold = number_freq.mean() + number_freq.std()
        strong_patterns = number_freq[number_freq > threshold].index.tolist()
        
        return {
            "strength": pattern_strength,
            "patterns": strong_patterns,
            "frequency_distribution": number_freq.to_dict(),
            "pattern_confidence": min(pattern_strength * 100, 95)
        }
    
    def calculate_prediction_risk_score(self, market, confidence, historical_accuracy):
        """Calculate risk score for predictions"""
        
        # Base risk from confidence (inverse relationship)
        confidence_risk = (100 - confidence) / 100
        
        # Historical accuracy risk
        accuracy_risk = (100 - historical_accuracy) / 100
        
        # Market-specific risk factors
        market_risk_factors = {
            "Time Bazar": 0.1,
            "Milan Day": 0.15,
            "Rajdhani Day": 0.12,
            "Kalyan": 0.08,
            "Milan Night": 0.18,
            "Rajdhani Night": 0.14,
            "Main Bazar": 0.1
        }
        
        market_risk = market_risk_factors.get(market, 0.15)
        
        # Combined risk score (0-1)
        total_risk = (confidence_risk * 0.4 + accuracy_risk * 0.4 + market_risk * 0.2)
        
        # Risk categories
        if total_risk < 0.3:
            risk_category = "LOW"
            recommendation = "STRONG BUY"
        elif total_risk < 0.5:
            risk_category = "MEDIUM"
            recommendation = "MODERATE BUY"
        elif total_risk < 0.7:
            risk_category = "HIGH"
            recommendation = "CAUTION"
        else:
            risk_category = "EXTREME"
            recommendation = "AVOID"
        
        return {
            "risk_score": total_risk,
            "risk_category": risk_category,
            "recommendation": recommendation,
            "confidence_factor": confidence_risk,
            "accuracy_factor": accuracy_risk,
            "market_factor": market_risk
        }
    
    def analyze_market_sentiment(self, df, market, window=30):
        """Analyze market sentiment based on recent trends"""
        market_data = df[df["Market"] == market].tail(window)
        
        if len(market_data) < 10:
            return {"sentiment": "NEUTRAL", "strength": 0}
        
        # Calculate trend indicators
        recent_opens = market_data["Open"].values
        recent_closes = market_data["Close"].values
        
        # Price momentum
        open_trend = np.polyfit(range(len(recent_opens)), recent_opens, 1)[0]
        close_trend = np.polyfit(range(len(recent_closes)), recent_closes, 1)[0]
        
        avg_trend = (open_trend + close_trend) / 2
        
        # Volatility trend
        volatility_window = 7
        volatilities = []
        for i in range(volatility_window, len(market_data)):
            segment = market_data.iloc[i-volatility_window:i]
            vol = segment["Open"].std() + segment["Close"].std()
            volatilities.append(vol)
        
        volatility_trend = np.polyfit(range(len(volatilities)), volatilities, 1)[0] if volatilities else 0
        
        # Sentiment calculation
        sentiment_score = 0
        
        # Trend factors
        if avg_trend > 0.1:
            sentiment_score += 0.3  # Positive trend
        elif avg_trend < -0.1:
            sentiment_score -= 0.3  # Negative trend
        
        # Volatility factors
        if volatility_trend < -0.05:
            sentiment_score += 0.2  # Decreasing volatility is good
        elif volatility_trend > 0.05:
            sentiment_score -= 0.2  # Increasing volatility is concerning
        
        # Pattern consistency
        pattern_analysis = self.analyze_pattern_strength(df, market)
        if pattern_analysis["strength"] > 0.3:
            sentiment_score += 0.2  # Strong patterns are positive
        
        # Determine sentiment
        if sentiment_score > 0.3:
            sentiment = "BULLISH"
        elif sentiment_score > 0.1:
            sentiment = "SLIGHTLY_BULLISH"
        elif sentiment_score > -0.1:
            sentiment = "NEUTRAL"
        elif sentiment_score > -0.3:
            sentiment = "SLIGHTLY_BEARISH"
        else:
            sentiment = "BEARISH"
        
        return {
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "strength": abs(sentiment_score),
            "trend_factor": avg_trend,
            "volatility_factor": volatility_trend,
            "pattern_factor": pattern_analysis["strength"]
        }
    
    def calculate_portfolio_risk(self, predictions_data):
        """Calculate overall portfolio risk across multiple predictions"""
        
        if not predictions_data:
            return {"portfolio_risk": 0, "diversification_score": 0}
        
        total_risk = 0
        market_count = len(predictions_data)
        
        # Calculate individual risks
        individual_risks = []
        for pred in predictions_data:
            market = pred.get("market", "")
            confidence = pred.get("confidence", 50)
            
            # Assume 75% historical accuracy if not provided
            historical_accuracy = pred.get("historical_accuracy", 75)
            
            risk_analysis = self.calculate_prediction_risk_score(
                market, confidence, historical_accuracy
            )
            individual_risks.append(risk_analysis["risk_score"])
        
        # Portfolio risk calculation
        avg_risk = np.mean(individual_risks)
        risk_std = np.std(individual_risks)
        
        # Diversification benefit (lower std = better diversification)
        diversification_score = max(0, 1 - risk_std)
        
        # Adjusted portfolio risk
        portfolio_risk = avg_risk * (1 - diversification_score * 0.2)
        
        return {
            "portfolio_risk": portfolio_risk,
            "average_individual_risk": avg_risk,
            "risk_standard_deviation": risk_std,
            "diversification_score": diversification_score,
            "total_markets": market_count,
            "risk_distribution": individual_risks
        }
    
    def generate_risk_report(self, df, market):
        """Generate comprehensive risk report for a market"""
        
        volatility_analysis = self.calculate_market_volatility(df, market)
        pattern_analysis = self.analyze_pattern_strength(df, market)
        sentiment_analysis = self.analyze_market_sentiment(df, market)
        
        # Overall risk assessment
        volatility_weight = 0.4
        pattern_weight = 0.3
        sentiment_weight = 0.3
        
        # Normalize scores (0-1, where 1 is highest risk)
        volatility_risk = min(volatility_analysis["volatility"] * 5, 1.0)
        pattern_risk = 1 - pattern_analysis["strength"]  # Low pattern strength = high risk
        sentiment_risk = 0.5 - (sentiment_analysis["sentiment_score"] * 0.5)  # Convert to risk
        
        overall_risk = (
            volatility_risk * volatility_weight +
            pattern_risk * pattern_weight +
            sentiment_risk * sentiment_weight
        )
        
        # Risk level
        if overall_risk < 0.3:
            overall_level = "LOW"
        elif overall_risk < 0.5:
            overall_level = "MEDIUM"
        elif overall_risk < 0.7:
            overall_level = "HIGH"
        else:
            overall_level = "EXTREME"
        
        return {
            "market": market,
            "overall_risk_score": overall_risk,
            "overall_risk_level": overall_level,
            "volatility_analysis": volatility_analysis,
            "pattern_analysis": pattern_analysis,
            "sentiment_analysis": sentiment_analysis,
            "recommendations": self._generate_recommendations(overall_risk, sentiment_analysis),
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, risk_score, sentiment_analysis):
        """Generate trading recommendations based on risk analysis"""
        
        recommendations = []
        
        # Risk-based recommendations
        if risk_score < 0.3:
            recommendations.append("✅ Low risk environment - Consider increasing position size")
            recommendations.append("📈 Good conditions for aggressive strategies")
        elif risk_score < 0.5:
            recommendations.append("⚠️ Moderate risk - Use standard position sizing")
            recommendations.append("📊 Monitor market closely for changes")
        elif risk_score < 0.7:
            recommendations.append("🚨 High risk detected - Reduce position sizes")
            recommendations.append("🛡️ Consider defensive strategies")
        else:
            recommendations.append("❌ Extreme risk - Avoid this market")
            recommendations.append("⏰ Wait for better conditions")
        
        # Sentiment-based recommendations
        sentiment = sentiment_analysis["sentiment"]
        if sentiment in ["BULLISH", "SLIGHTLY_BULLISH"]:
            recommendations.append("🔥 Positive sentiment - Look for buying opportunities")
        elif sentiment in ["BEARISH", "SLIGHTLY_BEARISH"]:
            recommendations.append("🐻 Negative sentiment - Exercise extra caution")
        else:
            recommendations.append("😐 Neutral sentiment - No strong directional bias")
        
        return recommendations
    
    def save_risk_analysis(self, risk_report):
        """Save risk analysis to database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_analysis (
                    id INTEGER PRIMARY KEY,
                    market TEXT,
                    risk_score REAL,
                    risk_level TEXT,
                    analysis_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert risk analysis
            cursor.execute('''
                INSERT INTO risk_analysis 
                (market, risk_score, risk_level, analysis_data) 
                VALUES (?, ?, ?, ?)
            ''', (
                risk_report["market"],
                risk_report["overall_risk_score"],
                risk_report["overall_risk_level"],
                json.dumps(risk_report)
            ))
            
            conn.commit()
            conn.close()
            
            print(f"Risk analysis saved for {risk_report['market']}")
            
        except Exception as e:
            print(f"Error saving risk analysis: {e}")

# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    # Load sample data
    try:
        df = pd.read_csv("satta_data.csv")
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        
        analyzer = RiskAnalyzer()
        
        # Generate risk report for a market
        report = analyzer.generate_risk_report(df, "Time Bazar")
        print(json.dumps(report, indent=2, default=str))
        
        # Save to database
        analyzer.save_risk_analysis(report)
        
    except Exception as e:
        print(f"Error: {e}")
