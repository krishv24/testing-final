# Teammate Setup Guide — IEEE Experiment Runner

This guide is for all team members (other than Krish) who want to
contribute runs to the shared experiment database.

---

## What you need

- Your own **Gemini API key** (free at https://aistudio.google.com/apikey)
- The **Supabase credentials** (ask Krish — these are shared for the whole team)
- Python 3.10+ installed
- The project code (cloned from GitHub)

---

## Step 1: Clone / pull the repo

```bash
git clone https://github.com/krishv24/testing-final.git
cd testing-final
git checkout secondaryllm-test
```

Or if you already have it:
```bash
git pull
git checkout secondaryllm-test
```

---

## Step 2: Set up virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

---

## Step 3: Create your `.env` file

Copy the example below and save it as `.env` in the project root
(same folder as `batch_runner.py`). Fill in **your own** Gemini key
and **your own name**.

```env
# GeoNames (use "demo" if you don't have one — it's rate limited but works)
GEONAMES_USERNAME=demo

# YOUR Gemini API key (get one free at https://aistudio.google.com/apikey)
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.5-flash-lite

# Supabase shared DB — ask Krish for these values
SUPABASE_URL=https://fgidyrkhzdiqvlffdnbn.supabase.co
SUPABASE_KEY=<ask Krish>

# YOUR name — this is how we track who ran what
RUNNER_NAME=yourname

# Leave these as-is
CACHE_DIR=.cache
```

> **Never commit your `.env` file** — it's already in `.gitignore`.

---

## Step 4: Test the connection

```bash
python batch_runner.py --dry-run
```

You should see something like:
```
=== DRY RUN ===
  [OK] Supabase connection OK
  [OK] 52 locations loaded
  [OK] 4 parameter combos per location
  [OK] Total possible runs: 208
  [OK] Already completed: X runs
  [OK] Remaining: Y runs
  [OK] Runner name: yourname
  [OK] Gemini model: gemini-2.5-flash-lite
```

---

## Step 5: Run experiments

```bash
# Run everything remaining (the DB tracks what's done — no duplicates)
python batch_runner.py

# Or run a specific number (good for testing first)
python batch_runner.py --limit 5

# Adjust delay between runs (default 5 seconds)
python batch_runner.py --delay 8
```

You can **stop and restart anytime** — the script checks Supabase
at the start and skips anything already completed. Multiple people can
run simultaneously without creating duplicates.

---

## Step 6: Check progress (optional)

```bash
python analyze_results.py --progress
```

---

## FAQ

**Q: The script says "quota exhausted" and stops — what do I do?**
It hit the free Gemini daily limit (~1500 requests/day, roughly
750 pipeline runs). Just run it again tomorrow:
```bash
python batch_runner.py
```
It will pick up exactly where it left off.

**Q: I got an error on one location and it skipped it — is the data lost?**
No. Error runs are stored in Supabase with the `error` field set,
so we can see what failed. They're not counted as "completed" so they
will be retried on the next run.

**Q: Can I run this on a lab computer overnight?**
Yes, just leave the terminal open:
```bash
python batch_runner.py --delay 10
```
It will run until all remaining experiments are done or the daily
Gemini quota is hit, then exit cleanly.

**Q: How do I know how many runs I've contributed?**
```bash
python analyze_results.py --progress
```
The summary shows runs broken down by `runner_name`.
