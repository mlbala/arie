-- Nothing wrote these tables: summaries were never generated, and answers are saved as
-- chat turns (arie_chat_turns). Databases created before 0.1.0 still have them.
DROP INDEX IF EXISTS idx_answers_paper_created;
DROP TABLE IF EXISTS answers;
DROP TABLE IF EXISTS briefs;
