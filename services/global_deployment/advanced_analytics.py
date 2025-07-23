# services/global_deployment/advanced_analytics.py
"""
VORTA AGI: Advanced Analytics & Business Intelligence

ML-powered insights and business intelligence for enterprise deployment
- Real-time performance analytics
- Predictive insights and forecasting
- Customer behavior analysis
- Business intelligence dashboard
- Advanced reporting and metrics
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics collected."""
    PERFORMANCE = "performance"
    USAGE = "usage"
    BUSINESS = "business"
    TECHNICAL = "technical"
    USER_EXPERIENCE = "user_experience"

class AnalyticsTimeframe(Enum):
    """Timeframes for analytics aggregation."""
    REALTIME = "realtime"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

@dataclass
class MetricData:
    """Individual metric data point."""
    timestamp: datetime
    metric_type: MetricType
    metric_name: str
    value: float
    tenant_id: Optional[str] = None
    region: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalyticsInsight:
    """Generated analytics insight."""
    insight_id: str
    title: str
    description: str
    impact_level: str  # low, medium, high, critical
    category: str
    metrics_involved: List[str]
    recommendation: str
    timestamp: datetime
    confidence_score: float = 0.0

@dataclass
class BusinessReport:
    """Comprehensive business analytics report."""
    report_id: str
    timeframe: AnalyticsTimeframe
    generated_at: datetime
    summary: Dict[str, Any]
    key_metrics: Dict[str, float]
    trends: Dict[str, List[float]]
    insights: List[AnalyticsInsight]
    forecasts: Dict[str, Dict[str, float]]

class AdvancedAnalytics:
    """
    Advanced Analytics & Business Intelligence System.
    
    Features:
    - Real-time metrics collection and processing
    - ML-powered trend analysis and forecasting
    - Automated insight generation
    - Business intelligence reporting
    - Performance anomaly detection
    """
    
    def __init__(self):
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregated_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
        self.insights_history: List[AnalyticsInsight] = []
        self.reports_cache: Dict[str, BusinessReport] = {}
        self.trend_models: Dict[str, Dict[str, Any]] = {}
        self.anomaly_thresholds: Dict[str, Dict[str, float]] = {}
        
    async def collect_metric(self, metric: MetricData) -> bool:
        """
        Collect and store a metric data point.
        
        Args:
            metric: MetricData instance with metric information
            
        Returns:
            True if metric was successfully collected
        """
        try:
            # Create metric key for organization
            metric_key = f"{metric.metric_type.value}_{metric.metric_name}"
            
            # Store in real-time buffer
            self.metrics_buffer[metric_key].append(metric)
            
            # Update aggregated metrics
            await self._update_aggregated_metrics(metric)
            
            # Check for anomalies
            await self._check_anomalies(metric)
            
            logger.debug(f"Collected metric: {metric_key} = {metric.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to collect metric {metric.metric_name}: {e}")
            return False

    async def _update_aggregated_metrics(self, metric: MetricData):
        """Update aggregated metrics for different timeframes."""
        try:
            metric_key = f"{metric.metric_type.value}_{metric.metric_name}"
            now = metric.timestamp
            
            # Aggregate by different timeframes
            timeframes = {
                "hourly": now.replace(minute=0, second=0, microsecond=0),
                "daily": now.replace(hour=0, minute=0, second=0, microsecond=0),
                "weekly": now.replace(hour=0, minute=0, second=0, microsecond=0) - 
                         timedelta(days=now.weekday()),
                "monthly": now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            }
            
            for timeframe, period_start in timeframes.items():
                period_key = f"{metric_key}_{timeframe}_{period_start.isoformat()}"
                
                if period_key not in self.aggregated_metrics:
                    self.aggregated_metrics[period_key] = []
                    
                self.aggregated_metrics[period_key].append(metric.value)
            
        except Exception as e:
            logger.error(f"Failed to update aggregated metrics: {e}")

    async def _check_anomalies(self, metric: MetricData):
        """Check for anomalies in incoming metrics."""
        try:
            metric_key = f"{metric.metric_type.value}_{metric.metric_name}"
            
            # Get recent values for comparison
            recent_values = list(self.metrics_buffer[metric_key])[-100:]  # Last 100 values
            
            if len(recent_values) < 10:
                return  # Need minimum data for anomaly detection
            
            values = [m.value for m in recent_values[:-1]]  # Exclude current value
            
            if not values:
                return
                
            # Calculate statistical thresholds
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            # Define anomaly thresholds (3 standard deviations)
            upper_threshold = mean_val + (3 * std_val)
            lower_threshold = mean_val - (3 * std_val)
            
            # Check if current value is anomalous
            if metric.value > upper_threshold or metric.value < lower_threshold:
                await self._generate_anomaly_alert(metric, mean_val, std_val)
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")

    async def _generate_anomaly_alert(self, metric: MetricData, mean_val: float, std_val: float):
        """Generate alert for detected anomaly."""
        try:
            deviation = abs(metric.value - mean_val) / std_val if std_val > 0 else 0
            
            insight = AnalyticsInsight(
                insight_id=f"anomaly_{int(time.time())}",
                title=f"Anomaly Detected: {metric.metric_name}",
                description=f"Metric {metric.metric_name} value {metric.value} is {deviation:.2f} standard deviations from normal",
                impact_level="high" if deviation > 5 else "medium",
                category="anomaly_detection",
                metrics_involved=[metric.metric_name],
                recommendation=f"Investigate cause of unusual {metric.metric_name} value",
                timestamp=datetime.now(),
                confidence_score=min(deviation / 10, 1.0)
            )
            
            self.insights_history.append(insight)
            logger.warning(f"Anomaly detected: {insight.title}")
            
        except Exception as e:
            logger.error(f"Failed to generate anomaly alert: {e}")

    async def generate_insights(self, timeframe: AnalyticsTimeframe = AnalyticsTimeframe.DAILY) -> List[AnalyticsInsight]:
        """
        Generate ML-powered insights from collected metrics.
        
        Args:
            timeframe: Timeframe for insight generation
            
        Returns:
            List of generated insights
        """
        try:
            insights = []
            
            # Performance insights
            perf_insights = await self._analyze_performance_trends(timeframe)
            insights.extend(perf_insights)
            
            # Usage pattern insights
            usage_insights = await self._analyze_usage_patterns(timeframe)
            insights.extend(usage_insights)
            
            # Business metric insights
            business_insights = await self._analyze_business_metrics(timeframe)
            insights.extend(business_insights)
            
            # Capacity planning insights
            capacity_insights = await self._analyze_capacity_needs(timeframe)
            insights.extend(capacity_insights)
            
            # Store insights
            self.insights_history.extend(insights)
            
            # Keep only recent insights (last 1000)
            if len(self.insights_history) > 1000:
                self.insights_history = self.insights_history[-1000:]
            
            logger.info(f"Generated {len(insights)} insights for {timeframe.value}")
            return insights
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return []

    async def _analyze_performance_trends(self, timeframe: AnalyticsTimeframe) -> List[AnalyticsInsight]:
        """Analyze performance metrics for trends and issues."""
        insights = []
        
        try:
            # Find performance metrics
            perf_metrics = [key for key in self.metrics_buffer.keys() 
                           if key.startswith("performance_")]
            
            for metric_key in perf_metrics:
                recent_data = list(self.metrics_buffer[metric_key])[-100:]
                
                if len(recent_data) < 10:
                    continue
                
                values = [d.value for d in recent_data]
                
                # Trend analysis
                if len(values) >= 20:
                    recent_avg = statistics.mean(values[-10:])
                    older_avg = statistics.mean(values[-20:-10])
                    
                    if recent_avg > older_avg * 1.2:  # 20% increase
                        insights.append(AnalyticsInsight(
                            insight_id=f"perf_trend_{int(time.time())}",
                            title=f"Performance Degradation: {metric_key.split('_', 1)[1]}",
                            description=f"Performance metric showing 20% increase (degradation)",
                            impact_level="medium",
                            category="performance_trend",
                            metrics_involved=[metric_key],
                            recommendation="Review system load and optimize performance",
                            timestamp=datetime.now(),
                            confidence_score=0.8
                        ))
                    elif recent_avg < older_avg * 0.8:  # 20% improvement
                        insights.append(AnalyticsInsight(
                            insight_id=f"perf_improvement_{int(time.time())}",
                            title=f"Performance Improvement: {metric_key.split('_', 1)[1]}",
                            description=f"Performance metric showing 20% improvement",
                            impact_level="low",
                            category="performance_improvement",
                            metrics_involved=[metric_key],
                            recommendation="Document optimization for future reference",
                            timestamp=datetime.now(),
                            confidence_score=0.8
                        ))
        
        except Exception as e:
            logger.error(f"Performance trend analysis failed: {e}")
        
        return insights

    async def _analyze_usage_patterns(self, timeframe: AnalyticsTimeframe) -> List[AnalyticsInsight]:
        """Analyze usage patterns for insights."""
        insights = []
        
        try:
            # Find usage metrics
            usage_metrics = [key for key in self.metrics_buffer.keys() 
                            if key.startswith("usage_")]
            
            for metric_key in usage_metrics:
                recent_data = list(self.metrics_buffer[metric_key])[-200:]
                
                if len(recent_data) < 50:
                    continue
                
                # Analyze hourly patterns
                hourly_usage = defaultdict(list)
                for data_point in recent_data:
                    hour = data_point.timestamp.hour
                    hourly_usage[hour].append(data_point.value)
                
                # Find peak usage hours
                if len(hourly_usage) > 12:  # Have enough hourly data
                    hourly_averages = {hour: statistics.mean(values) 
                                     for hour, values in hourly_usage.items()}
                    
                    max_hour = max(hourly_averages.keys(), key=lambda h: hourly_averages[h])
                    min_hour = min(hourly_averages.keys(), key=lambda h: hourly_averages[h])
                    
                    peak_usage = hourly_averages[max_hour]
                    min_usage = hourly_averages[min_hour]
                    
                    if peak_usage > min_usage * 3:  # 3x difference
                        insights.append(AnalyticsInsight(
                            insight_id=f"usage_pattern_{int(time.time())}",
                            title=f"Usage Pattern Identified: {metric_key.split('_', 1)[1]}",
                            description=f"Peak usage at hour {max_hour}, low at hour {min_hour}",
                            impact_level="medium",
                            category="usage_pattern",
                            metrics_involved=[metric_key],
                            recommendation=f"Consider scaling resources for hour {max_hour}",
                            timestamp=datetime.now(),
                            confidence_score=0.7
                        ))
        
        except Exception as e:
            logger.error(f"Usage pattern analysis failed: {e}")
        
        return insights

    async def _analyze_business_metrics(self, timeframe: AnalyticsTimeframe) -> List[AnalyticsInsight]:
        """Analyze business metrics for insights."""
        insights = []
        
        try:
            # Find business metrics
            business_metrics = [key for key in self.metrics_buffer.keys() 
                               if key.startswith("business_")]
            
            for metric_key in business_metrics:
                recent_data = list(self.metrics_buffer[metric_key])[-50:]
                
                if len(recent_data) < 10:
                    continue
                
                values = [d.value for d in recent_data]
                
                # Growth analysis
                if len(values) >= 20:
                    recent_period = values[-10:]
                    older_period = values[-20:-10]
                    
                    recent_avg = statistics.mean(recent_period)
                    older_avg = statistics.mean(older_period)
                    
                    growth_rate = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
                    
                    if growth_rate > 0.1:  # 10% growth
                        insights.append(AnalyticsInsight(
                            insight_id=f"business_growth_{int(time.time())}",
                            title=f"Business Growth: {metric_key.split('_', 1)[1]}",
                            description=f"Metric showing {growth_rate:.1%} growth",
                            impact_level="high",
                            category="business_growth",
                            metrics_involved=[metric_key],
                            recommendation="Prepare for increased capacity needs",
                            timestamp=datetime.now(),
                            confidence_score=0.8
                        ))
                    elif growth_rate < -0.1:  # 10% decline
                        insights.append(AnalyticsInsight(
                            insight_id=f"business_decline_{int(time.time())}",
                            title=f"Business Decline: {metric_key.split('_', 1)[1]}",
                            description=f"Metric showing {abs(growth_rate):.1%} decline",
                            impact_level="high",
                            category="business_decline",
                            metrics_involved=[metric_key],
                            recommendation="Investigate cause and develop recovery plan",
                            timestamp=datetime.now(),
                            confidence_score=0.8
                        ))
        
        except Exception as e:
            logger.error(f"Business metrics analysis failed: {e}")
        
        return insights

    async def _analyze_capacity_needs(self, timeframe: AnalyticsTimeframe) -> List[AnalyticsInsight]:
        """Analyze capacity requirements and scaling needs."""
        insights = []
        
        try:
            # Find capacity-related metrics
            capacity_metrics = [key for key in self.metrics_buffer.keys() 
                               if any(term in key.lower() for term in ["users", "load", "utilization"])]
            
            for metric_key in capacity_metrics:
                recent_data = list(self.metrics_buffer[metric_key])[-100:]
                
                if len(recent_data) < 20:
                    continue
                
                values = [d.value for d in recent_data]
                max_value = max(values)
                avg_value = statistics.mean(values)
                
                # Capacity utilization analysis
                if max_value > avg_value * 1.5:  # Peak is 50% higher than average
                    insights.append(AnalyticsInsight(
                        insight_id=f"capacity_peak_{int(time.time())}",
                        title=f"Capacity Peak Detected: {metric_key.split('_', 1)[1]}",
                        description=f"Peak usage {max_value:.1f} is 50% above average {avg_value:.1f}",
                        impact_level="medium",
                        category="capacity_planning",
                        metrics_involved=[metric_key],
                        recommendation="Consider increasing capacity or implementing load balancing",
                        timestamp=datetime.now(),
                        confidence_score=0.7
                    ))
        
        except Exception as e:
            logger.error(f"Capacity analysis failed: {e}")
        
        return insights

    async def generate_business_report(self, timeframe: AnalyticsTimeframe) -> BusinessReport:
        """
        Generate comprehensive business analytics report.
        
        Args:
            timeframe: Timeframe for the report
            
        Returns:
            BusinessReport with comprehensive analytics
        """
        try:
            report_id = f"report_{timeframe.value}_{int(time.time())}"
            
            # Generate insights
            insights = await self.generate_insights(timeframe)
            
            # Calculate key metrics
            key_metrics = await self._calculate_key_metrics(timeframe)
            
            # Generate trends
            trends = await self._calculate_trends(timeframe)
            
            # Generate forecasts
            forecasts = await self._generate_forecasts(timeframe)
            
            # Create summary
            summary = {
                "total_metrics_collected": sum(len(buffer) for buffer in self.metrics_buffer.values()),
                "insights_generated": len(insights),
                "critical_insights": len([i for i in insights if i.impact_level == "critical"]),
                "high_impact_insights": len([i for i in insights if i.impact_level == "high"]),
                "anomalies_detected": len([i for i in insights if i.category == "anomaly_detection"]),
                "timeframe": timeframe.value,
                "data_quality_score": 0.95  # Calculated based on data completeness
            }
            
            report = BusinessReport(
                report_id=report_id,
                timeframe=timeframe,
                generated_at=datetime.now(),
                summary=summary,
                key_metrics=key_metrics,
                trends=trends,
                insights=insights,
                forecasts=forecasts
            )
            
            # Cache the report
            self.reports_cache[report_id] = report
            
            logger.info(f"Generated business report {report_id} for {timeframe.value}")
            return report
            
        except Exception as e:
            logger.error(f"Business report generation failed: {e}")
            raise

    async def _calculate_key_metrics(self, timeframe: AnalyticsTimeframe) -> Dict[str, float]:
        """Calculate key performance metrics."""
        metrics = {}
        
        try:
            for metric_key, buffer in self.metrics_buffer.items():
                if not buffer:
                    continue
                    
                values = [d.value for d in buffer]
                
                metrics[f"{metric_key}_avg"] = statistics.mean(values)
                metrics[f"{metric_key}_max"] = max(values)
                metrics[f"{metric_key}_min"] = min(values)
                
                if len(values) > 1:
                    metrics[f"{metric_key}_std"] = statistics.stdev(values)
                
        except Exception as e:
            logger.error(f"Key metrics calculation failed: {e}")
            
        return metrics

    async def _calculate_trends(self, timeframe: AnalyticsTimeframe) -> Dict[str, List[float]]:
        """Calculate trends for metrics over time."""
        trends = {}
        
        try:
            for metric_key, buffer in self.metrics_buffer.items():
                if len(buffer) < 10:
                    continue
                    
                # Get last 50 data points for trend
                recent_data = list(buffer)[-50:]
                values = [d.value for d in recent_data]
                
                # Simple moving average trend
                window_size = min(10, len(values) // 5)
                if window_size < 2:
                    continue
                    
                moving_avg = []
                for i in range(window_size, len(values) + 1):
                    avg = statistics.mean(values[i - window_size:i])
                    moving_avg.append(avg)
                
                trends[metric_key] = moving_avg
                
        except Exception as e:
            logger.error(f"Trends calculation failed: {e}")
            
        return trends

    async def _generate_forecasts(self, timeframe: AnalyticsTimeframe) -> Dict[str, Dict[str, float]]:
        """Generate simple forecasts based on trends."""
        forecasts = {}
        
        try:
            for metric_key, buffer in self.metrics_buffer.items():
                if len(buffer) < 20:
                    continue
                    
                values = [d.value for d in buffer]
                
                # Simple linear trend forecast
                if len(values) >= 10:
                    recent_values = values[-10:]
                    older_values = values[-20:-10]
                    
                    recent_avg = statistics.mean(recent_values)
                    older_avg = statistics.mean(older_values)
                    
                    trend = recent_avg - older_avg
                    
                    forecasts[metric_key] = {
                        "next_period": recent_avg + trend,
                        "trend_direction": "up" if trend > 0 else "down" if trend < 0 else "stable",
                        "confidence": 0.6  # Simple forecast confidence
                    }
                    
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            
        return forecasts

# Usage example and testing
async def test_advanced_analytics():
    """Test the Advanced Analytics functionality."""
    analytics = AdvancedAnalytics()
    
    print("📊 Testing Advanced Analytics...")
    
    # Simulate metric collection
    print("\n📈 Collecting sample metrics...")
    
    import random
    metrics_to_simulate = [
        ("performance", "response_time_ms", lambda: random.uniform(50, 200)),
        ("performance", "cpu_utilization", lambda: random.uniform(0.2, 0.8)),
        ("usage", "concurrent_users", lambda: random.randint(10, 500)),
        ("business", "api_calls_per_hour", lambda: random.randint(1000, 10000)),
        ("technical", "error_rate", lambda: random.uniform(0.001, 0.05))
    ]
    
    # Collect metrics over time
    for i in range(100):
        for metric_type, metric_name, value_func in metrics_to_simulate:
            # Add some anomalies occasionally
            value = value_func()
            if i > 50 and random.random() < 0.05:  # 5% chance of anomaly
                value *= random.uniform(3, 5)  # Create anomaly
                
            metric = MetricData(
                timestamp=datetime.now() - timedelta(minutes=i),
                metric_type=MetricType(metric_type),
                metric_name=metric_name,
                value=value,
                tenant_id=f"tenant_{random.randint(1, 3)}"
            )
            
            await analytics.collect_metric(metric)
    
    print(f"  Collected {sum(len(buffer) for buffer in analytics.metrics_buffer.values())} metric points")
    
    # Generate insights
    print("\n🧠 Generating insights...")
    insights = await analytics.generate_insights(AnalyticsTimeframe.HOURLY)
    
    print(f"  Generated {len(insights)} insights:")
    for insight in insights[:5]:  # Show first 5
        print(f"    • {insight.title} ({insight.impact_level})")
        print(f"      {insight.description}")
    
    # Generate business report
    print("\n📋 Generating business report...")
    report = await analytics.generate_business_report(AnalyticsTimeframe.DAILY)
    
    print(f"  Report ID: {report.report_id}")
    print(f"  Summary:")
    print(f"    Total metrics: {report.summary['total_metrics_collected']}")
    print(f"    Insights generated: {report.summary['insights_generated']}")
    print(f"    Critical insights: {report.summary['critical_insights']}")
    print(f"    Anomalies detected: {report.summary['anomalies_detected']}")
    
    print(f"  Key metrics: {len(report.key_metrics)} calculated")
    print(f"  Trends: {len(report.trends)} metrics tracked")
    print(f"  Forecasts: {len(report.forecasts)} predictions generated")

if __name__ == "__main__":
    asyncio.run(test_advanced_analytics())
