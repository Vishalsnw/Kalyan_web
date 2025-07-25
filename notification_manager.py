
import os
import json
import sqlite3
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

class NotificationManager:
    def __init__(self):
        self.db_file = "satta_analytics.db"
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', 587)),
            'email': os.getenv('EMAIL_USER'),
            'password': os.getenv('EMAIL_PASS')
        }
        self.telegram_config = {
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'chat_id': os.getenv('TELEGRAM_CHAT_ID')
        }
        
    def send_email_notification(self, to_email, subject, body, is_html=False):
        """Send email notification"""
        try:
            if not all([self.email_config['email'], self.email_config['password']]):
                print("Email configuration missing")
                return False
                
            msg = MimeMultipart()
            msg['From'] = self.email_config['email']
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'html' if is_html else 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['email'], self.email_config['password'])
            text = msg.as_string()
            server.sendmail(self.email_config['email'], to_email, text)
            server.quit()
            
            print(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def send_telegram_notification(self, message):
        """Send Telegram notification"""
        try:
            if not all([self.telegram_config['bot_token'], self.telegram_config['chat_id']]):
                print("Telegram configuration missing")
                return False
                
            url = f"https://api.telegram.org/bot{self.telegram_config['bot_token']}/sendMessage"
            data = {
                'chat_id': self.telegram_config['chat_id'],
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data)
            if response.status_code == 200:
                print("Telegram notification sent successfully")
                return True
            else:
                print(f"Error sending Telegram notification: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error sending Telegram notification: {e}")
            return False
    
    def send_push_notification(self, title, body, user_tokens=None):
        """Send push notification (placeholder for future implementation)"""
        # This would integrate with services like Firebase Cloud Messaging
        print(f"Push notification: {title} - {body}")
        return True
    
    def notify_prediction_ready(self, market, confidence, predictions):
        """Notify when new predictions are ready"""
        subject = f"🎯 New AI Predictions for {market}"
        
        email_body = f"""
        <html>
        <body>
        <h2>🤖 Satta Matka Pro - AI Predictions</h2>
        <h3>Market: {market}</h3>
        <p><strong>AI Confidence:</strong> {confidence:.1f}%</p>
        
        <h4>Predictions:</h4>
        <ul>
            <li><strong>Open:</strong> {', '.join(map(str, predictions.get('open', [])))}</li>
            <li><strong>Close:</strong> {', '.join(map(str, predictions.get('close', [])))}</li>
            <li><strong>Jodis:</strong> {', '.join(map(str, predictions.get('jodis', [])[:5]))}</li>
        </ul>
        
        <p><em>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </body>
        </html>
        """
        
        telegram_msg = f"""
🎯 <b>New AI Predictions</b>

📊 <b>Market:</b> {market}
🎯 <b>Confidence:</b> {confidence:.1f}%

<b>Predictions:</b>
🔢 <b>Open:</b> {', '.join(map(str, predictions.get('open', [])))}
🔢 <b>Close:</b> {', '.join(map(str, predictions.get('close', [])))}
🎰 <b>Top Jodis:</b> {', '.join(map(str, predictions.get('jodis', [])[:5]))}

🕒 <i>Generated: {datetime.now().strftime('%H:%M')}</i>
        """
        
        # Send notifications
        self.send_telegram_notification(telegram_msg)
        
        # Get users who want prediction notifications
        users = self.get_notification_subscribers('predictions')
        for user_email in users:
            self.send_email_notification(user_email, subject, email_body, is_html=True)
    
    def notify_results_declared(self, market, actual_results, prediction_accuracy):
        """Notify when results are declared"""
        subject = f"🏆 Results Declared - {market}"
        
        email_body = f"""
        <html>
        <body>
        <h2>🏆 Satta Matka Pro - Results Declared</h2>
        <h3>Market: {market}</h3>
        
        <h4>Actual Results:</h4>
        <ul>
            <li><strong>Open:</strong> {actual_results.get('open')}</li>
            <li><strong>Close:</strong> {actual_results.get('close')}</li>
            <li><strong>Jodi:</strong> {actual_results.get('jodi')}</li>
        </ul>
        
        <h4>AI Prediction Accuracy:</h4>
        <ul>
            <li><strong>Open Match:</strong> {'✅ YES' if prediction_accuracy.get('open') else '❌ NO'}</li>
            <li><strong>Close Match:</strong> {'✅ YES' if prediction_accuracy.get('close') else '❌ NO'}</li>
            <li><strong>Jodi Match:</strong> {'✅ YES' if prediction_accuracy.get('jodi') else '❌ NO'}</li>
        </ul>
        
        <p><em>Declared at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </body>
        </html>
        """
        
        telegram_msg = f"""
🏆 <b>Results Declared</b>

📊 <b>Market:</b> {market}

<b>Actual Results:</b>
🔢 <b>Open:</b> {actual_results.get('open')}
🔢 <b>Close:</b> {actual_results.get('close')}
🎰 <b>Jodi:</b> {actual_results.get('jodi')}

<b>AI Accuracy:</b>
{'✅' if prediction_accuracy.get('open') else '❌'} Open Match
{'✅' if prediction_accuracy.get('close') else '❌'} Close Match  
{'✅' if prediction_accuracy.get('jodi') else '❌'} Jodi Match

🕒 <i>Declared: {datetime.now().strftime('%H:%M')}</i>
        """
        
        # Send notifications
        self.send_telegram_notification(telegram_msg)
        
        # Get users who want result notifications
        users = self.get_notification_subscribers('results')
        for user_email in users:
            self.send_email_notification(user_email, subject, email_body, is_html=True)
    
    def notify_high_confidence_prediction(self, market, confidence, predictions):
        """Send special notification for high confidence predictions"""
        if confidence >= 85:  # Only for very high confidence
            subject = f"🚨 HIGH CONFIDENCE Alert - {market} ({confidence:.1f}%)"
            
            telegram_msg = f"""
🚨 <b>HIGH CONFIDENCE ALERT</b> 🚨

📊 <b>Market:</b> {market}
🎯 <b>Confidence:</b> {confidence:.1f}%

⚡ <b>High Probability Predictions:</b>
🔢 <b>Open:</b> {', '.join(map(str, predictions.get('open', [])))}
🔢 <b>Close:</b> {', '.join(map(str, predictions.get('close', [])))}

🔥 <i>This is a high-confidence prediction!</i>
            """
            
            self.send_telegram_notification(telegram_msg)
    
    def get_notification_subscribers(self, notification_type):
        """Get list of users subscribed to specific notification type"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, notification_settings FROM user_preferences
            ''')
            
            subscribers = []
            for row in cursor.fetchall():
                user_id, settings_json = row
                if settings_json:
                    settings = json.loads(settings_json)
                    if settings.get(notification_type, False):
                        # In a real app, you'd map user_id to email
                        subscribers.append(f"{user_id}@example.com")
            
            conn.close()
            return subscribers
            
        except Exception as e:
            print(f"Error getting subscribers: {e}")
            return []
    
    def send_daily_summary(self):
        """Send daily performance summary"""
        try:
            # Get today's performance data
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            today = datetime.now().strftime('%Y-%m-%d')
            cursor.execute('''
                SELECT market, COUNT(*) as total, 
                       SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct
                FROM performance_tracking 
                WHERE date = ?
                GROUP BY market
            ''', (today,))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return
            
            summary = "📊 <b>Daily Performance Summary</b>\n\n"
            total_predictions = 0
            total_correct = 0
            
            for market, total, correct in results:
                accuracy = (correct / total * 100) if total > 0 else 0
                summary += f"📈 <b>{market}:</b> {correct}/{total} ({accuracy:.1f}%)\n"
                total_predictions += total
                total_correct += correct
            
            overall_accuracy = (total_correct / total_predictions * 100) if total_predictions > 0 else 0
            summary += f"\n🎯 <b>Overall:</b> {total_correct}/{total_predictions} ({overall_accuracy:.1f}%)"
            
            self.send_telegram_notification(summary)
            
        except Exception as e:
            print(f"Error sending daily summary: {e}")

# Example usage
if __name__ == "__main__":
    nm = NotificationManager()
    
    # Test notification
    test_predictions = {
        'open': [1, 5],
        'close': [3, 7],
        'jodis': ['15', '37', '28', '94', '06']
    }
    
    nm.notify_prediction_ready("Time Bazar", 87.5, test_predictions)
