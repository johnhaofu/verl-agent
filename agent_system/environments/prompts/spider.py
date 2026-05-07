# Spider text-to-SQL agent prompts.

SPIDER_TEMPLATE_NO_HIS = """
You are a SQL agent operating against a SQLite database.

Your task: write a SQL query that answers the user's question. You may
explore the database first by issuing exploratory SQL queries (which
return up to 10 rows or the SQLite error). When confident, commit your
final SQL via the `answer[...]` action.

Action format (pick exactly ONE per turn):
  <action>sql[SELECT ... FROM ...]</action>          (exploratory; returns rows or error)
  <action>answer[final SELECT statement]</action>    (commits and ends the episode)

ASCII only. Single SQL statement per action. No markdown fences.

Current state:
{current_observation}

Now reason step-by-step about what you need to discover or commit, then
emit one action.
You should first reason inside <think>...</think> tags about the schema,
the question, and what to query next. Then choose an action and present
it inside <action>...</action>.
"""

SPIDER_TEMPLATE = """
You are a SQL agent operating against a SQLite database.

Your task: write a SQL query that answers the user's question. You may
explore the database first by issuing exploratory SQL queries (which
return up to 10 rows or the SQLite error). When confident, commit your
final SQL via the `answer[...]` action.

Action format (pick exactly ONE per turn):
  <action>sql[SELECT ... FROM ...]</action>          (exploratory; returns rows or error)
  <action>answer[final SELECT statement]</action>    (commits and ends the episode)

ASCII only. Single SQL statement per action. No markdown fences.

Prior to this step, you have taken {step_count} step(s). Below are the
most recent {history_length} observations and the corresponding actions
you took:
{action_history}

You are now at step {current_step}. Your current observation is:
{current_observation}

Now reason step-by-step about what you've learned and what to do next.
Wrap your reasoning in <think>...</think> and your chosen action in
<action>...</action>.
"""
