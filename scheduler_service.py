
import schedule
import time
import threading
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import json
from notification_manager import NotificationManager
from risk_analyzer import RiskAnalyzer
from cache_manager import update_all_caches
import app  # Import the main app for accessing functions

class SchedulerService:
    def __init__(self):
        self.notification_manager = NotificationManager()
        self.risk_analyzer = RiskAnalyzer()
        self.is_running = False
        
    def start_scheduler(self):
        """Start the background scheduler"""
        if self.is_running:
            print("Scheduler is already running")
            return
        
        self.is_running = True
        
        # Schedule daily tasks
        schedule.every().day.at("09:00").do(self.daily_prediction_update)
        schedule.every().day.at("18:00").do(self.evening_results_check)
        schedule.every().day.at("22:00").do(self.daily_summary_report)
        
        # Schedule hourly tasks
        schedule.every().hour.do(self.update_cache)
        
        # Schedule frequent tasks
        schedule.every(30).minutes.do(self.auto_prediction_refresh)
        schedule.every(15).minutes.do(self.check_for_results)
        
        # Schedule weekly tasks
        schedule.every().monday.at("10:00").do(self.weekly_analysis_report)
        
        # Start scheduler in background thread
        def run_scheduler():
            print("🚀 Scheduler service started")
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        print("✅ Background scheduler initialized")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.is_running = False
        schedule.clear()
        print("🛑 Scheduler service stopped")
    
    def daily_prediction_update(self):
        """Generate and distribute daily predictions"""
        try:
            print("📊 Running daily prediction update...")
            
            # Load data
            df = app.load_data()
            prediction_date = app.next_prediction_date()
            
            predictions_summary = []
            
            for market in app.MARKETS:
                try:
                    open_vals, close_vals, jodis, status, confidence = app.train_and_predict_advanced(
                        df, market, prediction_date
                    )
                    
                    if open_vals and close_vals and jodis:
                        predictions = {
                            'open': [app.patti_to_digit(val) for val in open_vals],
                            'close': [app.patti_to_digit(val) for val in close_vals],
                            'jodis': jodis[:5]
                        }
                        
                        predictions_summary.append({
                            'market': market,
                            'confidence': confidence * 100,
                            'predictions': predictions
                        })
                        
                        # Send individual market notifications
                        self.notification_manager.notify_prediction_ready(
                            market, confidence * 100, predictions
                        )
                        
                        # Check for high confidence predictions
                        if confidence >= 0.85:
                            self.notification_manager.notify_high_confidence_prediction(
                                market, confidence * 100, predictions
                            )
                        
                        # Generate and save risk analysis
                        risk_report = self.risk_analyzer.generate_risk_report(df, market)
                        self.risk_analyzer.save_risk_analysis(risk_report)
                
                except Exception as e:
                    print(f"Error processing {market}: {e}")
            
            # Send consolidated daily report
            if predictions_summary:
                self.send_daily_predictions_report(predictions_summary, prediction_date)
            
            print(f"✅ Daily prediction update completed for {len(predictions_summary)} markets")
            
        except Exception as e:
            print(f"❌ Error in daily prediction update: {e}")
    
    def evening_results_check(self):
        """Check and process evening results"""
        try:
            print("🔍 Running evening results check...")
            
            # This would integrate with result checking service
            self.check_for_results()
            
            # Generate accuracy report
            self.generate_accuracy_report()
            
            print("✅ Evening results check completed")
            
        except Exception as e:
            print(f"❌ Error in evening results check: {e}")
    
    def check_for_results(self):
        """Check for new results and notify users"""
        try:
            # This would integrate with satta_result_checker.py
            # For now, we'll check if there are new results in the database
            
            today = datetime.now().strftime("%d/%m/%Y")
            df = app.load_data()
            today_results = df[df["Date"].dt.strftime("%d/%m/%Y") == today]
            
            if not today_results.empty:
                print(f"📈 Found {len(today_results)} results for today")
                
                # Check prediction accuracy for today's results
                self.check_prediction_accuracy(today_results)
            
        except Exception as e:
            print(f"❌ Error checking for results: {e}")
    
    def check_prediction_accuracy(self, results_df):
        """Check accuracy of today's predictions against actual results"""
        try:
            # Load today's predictions
            pred_df = pd.read_csv(app.PRED_FILE)
            pred_df['Date'] = pd.to_datetime(pred_df['Date'], dayfirst=True, errors='coerce').dt.strftime('%d/%m/%Y')
            today = datetime.now().strftime("%d/%m/%Y")
            today_preds = pred_df[pred_df['Date'] == today]
            
            for _, result in results_df.iterrows():
                market = result['Market']
                pred_row = today_preds[today_preds['Market'] == market]
                
                if not pred_row.empty:
                    prediction = pred_row.iloc[0]
                    
                    # Check matches
                    pred_open = prediction['Open'].split(', ')
                    pred_close = prediction['Close'].split(', ')
                    pred_jodis = prediction['Jodis'].split(', ')
                    
                    actual_jodi = str(result['Jodi']).zfill(2)
                    actual_open_digit = actual_jodi[0]
                    actual_close_digit = actual_jodi[1]
                    
                    accuracy = {
                        'open': actual_open_digit in pred_open,
                        'close': actual_close_digit in pred_close,
                        'jodi': actual_jodi in pred_jodis
                    }
                    
                    actual_results = {
                        'open': result['Open'],
                        'close': result['Close'],
                        'jodi': result['Jodi']
                    }
                    
                    # Notify about results
                    self.notification_manager.notify_results_declared(
                        market, actual_results, accuracy
                    )
                    
                    # Save to performance tracking
                    self.save_performance_data(market, today, prediction, result, accuracy)
        
        except Exception as e:
            print(f"❌ Error checking prediction accuracy: {e}")
    
    def save_performance_data(self, market, date, prediction, actual, accuracy):
        """Save performance data to database"""
        try:
            conn = sqlite3.connect(app.DB_FILE)
            cursor = conn.cursor()
            
            # Save performance for each prediction type
            performance_data = [
                (market, date, 'open', prediction['Open'], str(actual['Open']), accuracy['open']),
                (market, date, 'close', prediction['Close'], str(actual['Close']), accuracy['close']),
                (market, date, 'jodi', prediction['Jodis'], str(actual['Jodi']), accuracy['jodi'])
            ]
            
            for data in performance_data:
                cursor.execute('''
                    INSERT INTO performance_tracking 
                    (market, date, prediction_type, predicted_value, actual_value, is_correct) 
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', data)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error saving performance data: {e}")
    
    def auto_prediction_refresh(self):
        """Auto-refresh predictions with latest data"""
        try:
            print("🔄 Auto-refreshing predictions...")
            
            # Update cache
            update_all_caches()
            
            # Check if we need to generate new predictions
            df = app.load_data()
            last_update = df["Date"].max() if not df.empty else datetime.now() - timedelta(days=1)
            
            # If data is recent, refresh predictions
            if datetime.now() - last_update < timedelta(hours=2):
                print("📊 Generating fresh predictions...")
                # This would trigger prediction regeneration
            
        except Exception as e:
            print(f"❌ Error in auto prediction refresh: {e}")
    
    def update_cache(self):
        """Update all cache files"""
        try:
            print("💾 Updating cache files...")
            update_all_caches()
            print("✅ Cache files updated")
        except Exception as e:
            print(f"❌ Error updating cache: {e}")
    
    def daily_summary_report(self):
        """Generate and send daily summary report"""
        try:
            print("📋 Generating daily summary report...")
            self.notification_manager.send_daily_summary()
            print("✅ Daily summary report sent")
        except Exception as e:
            print(f"❌ Error generating daily summary: {e}")
    
    def weekly_analysis_report(self):
        """Generate weekly analysis and insights"""
        try:
            print("📊 Generating weekly analysis report...")
            
            # Calculate weekly performance
            week_start = datetime.now() - timedelta(days=7)
            week_end = datetime.now()
            
            conn = sqlite3.connect(app.DB_FILE)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT market, 
                       COUNT(*) as total_predictions,
                       SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
                       AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) * 100 as accuracy
                FROM performance_tracking 
                WHERE date BETWEEN ? AND ?
                GROUP BY market
                ORDER BY accuracy DESC
            ''', (week_start.strftime('%Y-%m-%d'), week_end.strftime('%Y-%m-%d')))
            
            results = cursor.fetchall()
            conn.close()
            
            if results:
                report = "📈 <b>Weekly Performance Report</b>\n\n"
                
                for market, total, correct, accuracy in results:
                    report += f"📊 <b>{market}:</b>\n"
                    report += f"   • Predictions: {total}\n"
                    report += f"   • Correct: {correct}\n"
                    report += f"   • Accuracy: {accuracy:.1f}%\n\n"
                
                # Find best and worst performing markets
                best_market = results[0] if results else None
                worst_market = results[-1] if results else None
                
                if best_market:
                    report += f"🏆 <b>Best Market:</b> {best_market[0]} ({best_market[3]:.1f}%)\n"
                if worst_market and len(results) > 1:
                    report += f"⚠️ <b>Needs Attention:</b> {worst_market[0]} ({worst_market[3]:.1f}%)\n"
                
                # Send report
                self.notification_manager.send_telegram_notification(report)
            
            print("✅ Weekly analysis report generated")
            
        except Exception as e:
            print(f"❌ Error generating weekly report: {e}")
    
    def generate_accuracy_report(self):
        """Generate accuracy report for today"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            conn = sqlite3.connect(app.DB_FILE)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT market, prediction_type,
                       COUNT(*) as total,
                       SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct
                FROM performance_tracking 
                WHERE date = ?
                GROUP BY market, prediction_type
            ''', (today,))
            
            results = cursor.fetchall()
            conn.close()
            
            if results:
                print(f"📊 Today's Accuracy Report:")
                for market, pred_type, total, correct in results:
                    accuracy = (correct / total * 100) if total > 0 else 0
                    print(f"   {market} - {pred_type}: {correct}/{total} ({accuracy:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error generating accuracy report: {e}")
    
    def send_daily_predictions_report(self, predictions_summary, date):
        """Send consolidated daily predictions report"""
        try:
            report = f"🤖 <b>AI Predictions Summary - {date}</b>\n\n"
            
            total_markets = len(predictions_summary)
            avg_confidence = sum(p['confidence'] for p in predictions_summary) / total_markets
            
            report += f"📊 <b>Markets Analyzed:</b> {total_markets}\n"
            report += f"🎯 <b>Average Confidence:</b> {avg_confidence:.1f}%\n\n"
            
            # Sort by confidence
            predictions_summary.sort(key=lambda x: x['confidence'], reverse=True)
            
            for pred in predictions_summary:
                market = pred['market']
                confidence = pred['confidence']
                predictions = pred['predictions']
                
                report += f"📈 <b>{market}</b> ({confidence:.1f}%)\n"
                report += f"   🔢 Open: {', '.join(map(str, predictions['open']))}\n"
                report += f"   🔢 Close: {', '.join(map(str, predictions['close']))}\n"
                report += f"   🎰 Jodis: {', '.join(predictions['jodis'][:3])}\n\n"
            
            report += f"🕒 <i>Generated at: {datetime.now().strftime('%H:%M')}</i>"
            
            # Send via Telegram
            self.notification_manager.send_telegram_notification(report)
            
        except Exception as e:
            print(f"❌ Error sending daily predictions report: {e}")

# Global scheduler instance
scheduler_service = None

def start_background_scheduler():
    """Start the background scheduler service"""
    global scheduler_service
    if scheduler_service is None:
        scheduler_service = SchedulerService()
        scheduler_service.start_scheduler()
    return scheduler_service

def stop_background_scheduler():
    """Stop the background scheduler service"""
    global scheduler_service
    if scheduler_service:
        scheduler_service.stop_scheduler()
        scheduler_service = None

if __name__ == "__main__":
    # Test the scheduler
    service = SchedulerService()
    service.start_scheduler()
    
    try:
        # Keep running
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n🛑 Stopping scheduler...")
        service.stop_scheduler()
