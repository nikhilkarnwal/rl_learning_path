# Fix for Installation Issues

The installation errors are due to incomplete Xcode Command Line Tools installation. Here's how to fix it:

## Option 1: Use Homebrew Python (Recommended - Fastest)

1. **Remove the old virtual environment:**
   ```bash
   cd /Users/nikarn/Desktop/projects/MLStudy/rl_learning_path
   rm -rf rlenv
   ```

2. **Create new virtual environment with Homebrew Python:**
   ```bash
   /opt/homebrew/bin/python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install requirements:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements-simple.txt
   ```

4. **Test the setup:**
   ```bash
   python src/main.py --test-env
   ```

## Option 2: Fix Command Line Tools (More time consuming)

If you need Box2D environments (LunarLander), you'll need to properly install Command Line Tools:

1. **Check if the installation dialog is still open** - complete it if it's waiting
2. **Or reinstall manually:**
   ```bash
   sudo rm -rf /Library/Developer/CommandLineTools
   xcode-select --install
   ```
3. **Wait for installation to complete** (can take 5-10 minutes)
4. **Then use Option 1 steps above**

## Quick Start (Recommended)

Just run these commands:
```bash
cd /Users/nikarn/Desktop/projects/MLStudy/rl_learning_path
rm -rf rlenv
/opt/homebrew/bin/python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-simple.txt
python src/main.py --test-env
```
