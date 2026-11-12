"""Known Gregorian day counts; no sales, holiday or business-day assumptions."""

from calendar import monthrange
import re

COLUMNS = ['days_in_month'] + [f'weekday_count_{i}' for i in range(7)]
DEFINITION = {'unit': 'calendar_days_per_gregorian_month',
              'value_type': 'DETERMINISTIC_KNOWN_CALENDAR', 'weekday_zero': 'Monday',
              'applies_to': 'all_seven_markets_without_country_holiday_assumptions',
              'target_month': 'forecast_target_month', 'availability': 'deterministic_before_origin',
              'missing_policy': 'invalid_month_is_error', 'sales_unit_changed': False}


def month_features(month):
    if not isinstance(month, str) or re.fullmatch(r'[0-9]{4}-[0-9]{2}', month) is None:
        raise ValueError('Calendar month must be YYYY-MM')
    year, number = map(int, month.split('-'))
    if not 1 <= year <= 9999 or not 1 <= number <= 12:
        raise ValueError('Invalid calendar month')
    first_weekday, days = monthrange(year, number)
    weeks, remaining = divmod(days, 7)
    return {'days_in_month': days,
            **{f'weekday_count_{weekday}': weeks + int((weekday - first_weekday) % 7 < remaining)
               for weekday in range(7)}}
