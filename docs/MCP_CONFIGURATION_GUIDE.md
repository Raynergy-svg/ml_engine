# MCP Configuration Guide - ML Engine Control Center

## Overview

The [`.kilocode/mcp.json`](../.kilocode/mcp.json) file configures a comprehensive Model Context Protocol (MCP) control center for all AI and ML tasks in the ML Engine project. This configuration integrates 10 specialized servers that provide capabilities for the entire machine learning lifecycle, from data preprocessing to deployment.

## Configuration Structure

```json
{
  "mcpServers": {
    "server-name": {
      "command": "npx",
      "args": ["-y", "@package-name", "additional-args"],
      "description": "Server description"
    }
  }
}
```

## Configured MCP Servers

### 1. Filesystem Server
**Package:** `@modelcontextprotocol/server-filesystem`

**Purpose:** Provides complete file system access for managing project files, data, models, notebooks, and configuration files.

**Capabilities:**
- Read, write, and organize ML artifacts across the entire project directory
- Access training data (`market_data/`)
- Manage model files (`trained_data/models/`)
- Edit configuration files (`config/`)
- Work with notebooks (`notebooks/`)
- Organize documentation (`docs/`)

**Use Cases:**
- Loading training data from CSV files
- Saving trained models to disk
- Reading/writing YAML configuration files
- Managing experiment logs and outputs
- Organizing feature engineering scripts

---

### 2. Git Server
**Package:** `@modelcontextprotocol/server-git`

**Purpose:** Git version control integration for tracking model experiments, managing code branches, and maintaining reproducible workflows.

**Capabilities:**
- Track model experiments with version control
- Manage code branches for different trading strategies
- Collaborate on ML projects
- Maintain reproducible workflows
- Roll back to previous model versions

**Use Cases:**
- Commit model checkpoints with training metadata
- Create branches for experimental features
- Track hyperparameter tuning experiments
- Merge successful strategies into main branch
- Review code changes before deployment

---

### 3. Brave Search Server
**Package:** `@modelcontextprotocol/server-brave-search`

**Purpose:** Web search capability for finding latest research papers, documentation, tutorials, best practices, and troubleshooting solutions.

**Capabilities:**
- Search for latest ML research papers
- Find documentation for TensorFlow, PyTorch, XGBoost, LightGBM
- Discover tutorials and best practices
- Troubleshoot common ML issues
- Stay updated with framework releases

**Use Cases:**
- Finding papers on transformer architectures for time series
- Searching for XGBoost hyperparameter tuning guides
- Discovering new techniques for feature engineering
- Troubleshooting TensorFlow Metal GPU issues
- Finding best practices for ensemble models

---

### 4. Fetch Server
**Package:** `@modelcontextprotocol/server-fetch`

**Purpose:** HTTP client for retrieving real-time documentation from framework websites, API references, and fetching data from external sources.

**Capabilities:**
- Retrieve documentation from TensorFlow.org, PyTorch.org, scikit-learn.org
- Access XGBoost and LightGBM API references
- Fetch data from external APIs
- Download datasets from remote sources
- Access real-time market data feeds

**Use Cases:**
- Fetching latest TensorFlow API documentation
- Downloading financial datasets from APIs
- Retrieving model weights from remote servers
- Accessing OANDA trading API documentation
- Fetching research paper PDFs

---

### 5. Puppeteer Server
**Package:** `@modelcontextprotocol/server-puppeteer`

**Purpose:** Web scraping and automation for extracting research papers from arXiv, documentation from dynamic websites, monitoring ML conference proceedings, and collecting market data.

**Capabilities:**
- Extract research papers from arXiv
- Scrape documentation from dynamic websites
- Monitor ML conference proceedings (NeurIPS, ICML, ICLR)
- Collect market data for trading models
- Automate web interactions for data collection

**Use Cases:**
- Scraping arXiv for latest time series forecasting papers
- Extracting trading signals from financial news websites
- Monitoring conference schedules and proceedings
- Collecting historical price data from trading platforms
- Automating form submissions for data collection

---

### 6. GitHub Server
**Package:** `@modelcontextprotocol/server-github`

**Purpose:** GitHub integration for accessing framework repositories, exploring code examples, managing model releases, and collaborating with the ML community.

**Capabilities:**
- Access framework repositories (tensorflow/tensorflow, pytorch/pytorch, dmlc/xgboost)
- Explore code examples and implementations
- Manage model releases and versioning
- Collaborate with the ML community
- Track issues and pull requests

**Use Cases:**
- Exploring TensorFlow transformer implementations
- Finding XGBoost best practices from official repo
- Contributing to open-source ML projects
- Tracking bug fixes and feature requests
- Accessing community-maintained model zoos

---

### 7. Memory Server
**Package:** `@modelcontextprotocol/server-memory`

**Purpose:** Persistent memory storage for maintaining experiment results, model metadata, training configurations, hyperparameter tuning history, and contextual information across ML workflows.

**Capabilities:**
- Store experiment results persistently
- Maintain model metadata and versioning
- Track training configurations
- Record hyperparameter tuning history
- Preserve contextual information across sessions

**Use Cases:**
- Storing training metrics across multiple runs
- Remembering successful hyperparameter combinations
- Tracking model performance over time
- Maintaining context for long-running experiments
- Storing feature engineering insights

---

### 8. Sequential Thinking Server
**Package:** `@modelcontextprotocol/server-sequential-thinking`

**Purpose:** Reasoning engine for complex ML problem-solving, architectural decisions, debugging model issues, designing experiments, and systematic analysis of trading strategies.

**Capabilities:**
- Complex ML problem-solving
- Architectural decision-making
- Systematic debugging of model issues
- Designing structured experiments
- Analyzing trading strategies systematically

**Use Cases:**
- Designing ensemble model architectures
- Debugging gradient vanishing in transformers
- Planning systematic hyperparameter searches
- Analyzing why a trading strategy fails
- Designing ablation studies

---

### 9. E2B Code Interpreter
**Package:** `@e2b/code-interpreter`

**Purpose:** Isolated Python code execution environment for safe data analysis, model training, feature engineering, statistical testing, and running ML experiments without affecting the local environment.

**Capabilities:**
- Execute Python code in isolated environment
- Run data analysis safely
- Train models in sandbox
- Perform statistical testing
- Experiment with code without risk

**Use Cases:**
- Testing new feature engineering techniques
- Running quick data exploration
- Training small experimental models
- Testing code snippets before integration
- Running statistical significance tests

---

### 10. PostgreSQL Server
**Package:** `@modelcontextprotocol/server-postgres`

**Purpose:** Structured database integration for storing experiment metadata, model performance metrics, trading signals, backtest results, and enabling advanced analytics queries.

**Capabilities:**
- Store structured experiment metadata
- Track model performance metrics
- Log trading signals and executions
- Store backtest results
- Enable complex analytics queries

**Use Cases:**
- Querying training metrics across experiments
- Analyzing historical trading signal performance
- Comparing model versions with SQL
- Storing and retrieving feature importance data
- Running analytics on backtest results

---

## ML Lifecycle Coverage

### Data Preprocessing Phase

**Servers Used:**
- **Filesystem**: Load raw data from `market_data/`
- **E2B Code Interpreter**: Perform exploratory data analysis
- **PostgreSQL**: Store processed features
- **Sequential Thinking**: Design preprocessing pipeline

**Workflow:**
1. Load raw market data using Filesystem
2. Explore data distribution in E2B environment
3. Design feature engineering strategy with Sequential Thinking
4. Implement and test preprocessing code
5. Store processed features in PostgreSQL

---

### Model Development Phase

**Servers Used:**
- **Brave Search**: Find latest research papers
- **Fetch**: Retrieve framework documentation
- **GitHub**: Explore code examples
- **E2B Code Interpreter**: Test model architectures
- **Memory**: Track experiment configurations
- **Sequential Thinking**: Design model architecture

**Workflow:**
1. Search for relevant research papers
2. Study framework documentation
3. Explore implementations on GitHub
4. Design model architecture systematically
5. Test architectures in isolated environment
6. Track successful configurations in Memory

---

### Model Training Phase

**Servers Used:**
- **Filesystem**: Save model checkpoints
- **PostgreSQL**: Log training metrics
- **Memory**: Store hyperparameter results
- **Git**: Version control model artifacts
- **E2B Code Interpreter**: Run training experiments

**Workflow:**
1. Configure training parameters
2. Run training in E2B environment
3. Log metrics to PostgreSQL
4. Save model checkpoints to Filesystem
5. Track hyperparameter results in Memory
6. Commit successful models to Git

---

### Model Evaluation Phase

**Servers Used:**
- **E2B Code Interpreter**: Run evaluation scripts
- **PostgreSQL**: Query performance metrics
- **Sequential Thinking**: Analyze results systematically
- **Memory**: Store evaluation insights

**Workflow:**
1. Run evaluation in isolated environment
2. Query historical metrics from PostgreSQL
3. Analyze results systematically
4. Store insights in Memory
5. Compare with baseline models

---

### Deployment Phase

**Servers Used:**
- **Git**: Manage deployment branches
- **Filesystem**: Prepare production artifacts
- **PostgreSQL**: Monitor production metrics
- **Fetch**: Access production APIs

**Workflow:**
1. Create deployment branch in Git
2. Prepare model artifacts in Filesystem
3. Deploy to production environment
4. Monitor metrics in PostgreSQL
5. Roll back if needed using Git

---

## Framework-Specific Capabilities

### TensorFlow
- **Documentation**: Fetch server retrieves from tensorflow.org
- **Code Examples**: GitHub server accesses tensorflow/tensorflow
- **Research**: Brave Search finds TensorFlow papers
- **Training**: E2B Code Interpreter runs TF training
- **Metal Support**: Sequential Thinking helps debug GPU issues

### PyTorch
- **Documentation**: Fetch server retrieves from pytorch.org
- **Code Examples**: GitHub server accesses pytorch/pytorch
- **Research**: Brave Search finds PyTorch papers
- **Training**: E2B Code Interpreter runs PyTorch training

### XGBoost
- **Documentation**: Fetch server retrieves XGBoost docs
- **Code Examples**: GitHub server accesses dmlc/xgboost
- **Research**: Brave Search finds XGBoost best practices
- **Training**: E2B Code Interpreter runs XGBoost training

### LightGBM
- **Documentation**: Fetch server retrieves LightGBM docs
- **Code Examples**: GitHub server accesses microsoft/LightGBM
- **Research**: Brave Search finds LightGBM papers
- **Training**: E2B Code Interpreter runs LightGBM training

### scikit-learn
- **Documentation**: Fetch server retrieves from scikit-learn.org
- **Code Examples**: GitHub server accesses scikit-learn/scikit-learn
- **Research**: Brave Search finds sklearn best practices
- **Training**: E2B Code Interpreter runs sklearn models

---

## Trading-Specific Workflows

### Market Data Collection
1. **Puppeteer**: Scrape market data from financial websites
2. **Fetch**: Access OANDA API for real-time data
3. **Filesystem**: Store data in `market_data/`
4. **PostgreSQL**: Log data collection metadata

### Feature Engineering
1. **Sequential Thinking**: Design feature strategy
2. **E2B Code Interpreter**: Test feature extraction
3. **Filesystem**: Save feature engineering scripts
4. **Memory**: Store successful feature combinations

### Model Training for Trading
1. **Brave Search**: Find latest time series forecasting papers
2. **GitHub**: Explore trading model implementations
3. **E2B Code Interpreter**: Train models safely
4. **PostgreSQL**: Log training metrics
5. **Memory**: Track hyperparameter results
6. **Git**: Version control model artifacts

### Backtesting
1. **E2B Code Interpreter**: Run backtest simulations
2. **PostgreSQL**: Store backtest results
3. **Sequential Thinking**: Analyze performance systematically
4. **Memory**: Store successful strategies

### Deployment to Production
1. **Git**: Create deployment branch
2. **Filesystem**: Prepare production artifacts
3. **PostgreSQL**: Monitor production metrics
4. **Fetch**: Access production trading APIs

---

## Example Workflows

### Workflow 1: Research New Model Architecture

```bash
# 1. Search for latest research
mcp--brave-search--search "transformer time series forecasting 2024"

# 2. Fetch paper from arXiv
mcp--fetch--fetch "https://arxiv.org/abs/xxxx.xxxxx"

# 3. Explore implementation on GitHub
mcp--github--search_repositories "transformer time series"

# 4. Design architecture systematically
mcp--sequential-thinking--sequentialthinking "Design transformer for FX trading"

# 5. Test in isolated environment
mcp--e2b-code-interpreter--execute "python test_transformer.py"

# 6. Store insights
mcp--memory--create_entities "transformer insights"
```

### Workflow 2: Train and Evaluate Model

```bash
# 1. Load data
mcp--filesystem--read_file "market_data/EUR_USD_H1.csv"

# 2. Explore data
mcp--e2b-code-interpreter--execute "import pandas as pd; df = pd.read_csv('market_data/EUR_USD_H1.csv'); df.describe()"

# 3. Train model
mcp--e2b-code-interpreter--execute "python train_model.py"

# 4. Log metrics
mcp--postgres--query "INSERT INTO training_metrics VALUES (...)"

# 5. Save model
mcp--filesystem--write_file "trained_data/models/model.pkl" "model_data"

# 6. Track experiment
mcp--memory--create_entities "experiment_001"

# 7. Commit to Git
mcp--git--commit "Add trained model for EUR_USD"
```

### Workflow 3: Debug Training Issues

```bash
# 1. Analyze problem systematically
mcp--sequential-thinking--sequentialthinking "Debug gradient vanishing in transformer"

# 2. Search for solutions
mcp--brave-search--search "transformer gradient vanishing fix"

# 3. Test fix in isolation
mcp--e2b-code-interpreter--execute "python test_fix.py"

# 4. Query previous experiments
mcp--postgres--query "SELECT * FROM training_metrics WHERE accuracy < 0.5"

# 5. Review similar issues
mcp--github--search_issues "gradient vanishing tensorflow"

# 6. Document solution
mcp--memory--create_entities "gradient_vanishing_solution"
```

---

## Installation and Setup

### Prerequisites
- Node.js and npm installed
- PostgreSQL running on localhost:5432
- Git initialized in project directory

### Installation Steps

1. **Install MCP servers:**
   ```bash
   npx -y @modelcontextprotocol/server-filesystem /Users/davidcertan/Desktop/ml_engine
   npx -y @modelcontextprotocol/server-git --repository /Users/davidcertan/Desktop/ml_engine
   npx -y @modelcontextprotocol/server-brave-search
   npx -y @modelcontextprotocol/server-fetch
   npx -y @modelcontextprotocol/server-puppeteer
   npx -y @modelcontextprotocol/server-github
   npx -y @modelcontextprotocol/server-memory
   npx -y @modelcontextprotocol/server-sequential-thinking
   npx -y @e2b/code-interpreter
   npx -y @modelcontextprotocol/server-postgres postgresql://localhost:5432/ml_engine
   ```

2. **Configure PostgreSQL database:**
   ```bash
   createdb ml_engine
   ```

3. **Verify configuration:**
   ```bash
   cat .kilocode/mcp.json
   ```

---

## Troubleshooting

### Server Connection Issues
- **Problem**: MCP server not starting
- **Solution**: Check npm is installed, verify network connection, ensure PostgreSQL is running

### Filesystem Access Denied
- **Problem**: Cannot access project files
- **Solution**: Verify path in configuration matches actual project location

### PostgreSQL Connection Failed
- **Problem**: Cannot connect to database
- **Solution**: Ensure PostgreSQL is running, verify connection string, check database exists

### E2B Code Interpreter Not Working
- **Problem**: Code execution fails
- **Solution**: Check E2B API key, verify internet connection, ensure sufficient credits

### Puppeteer Scraping Errors
- **Problem**: Web scraping fails
- **Solution**: Check website accessibility, verify selectors, handle dynamic content properly

---

## Best Practices

### 1. Use Isolated Environments
Always use the E2B Code Interpreter for experimental code to avoid affecting the local environment.

### 2. Track All Experiments
Use the Memory server to store experiment configurations, results, and insights for reproducibility.

### 3. Version Control Everything
Commit model artifacts, configurations, and training scripts to Git for full reproducibility.

### 4. Log Metrics to Database
Store all training and evaluation metrics in PostgreSQL for easy querying and analysis.

### 5. Document Decisions
Use Sequential Thinking to systematically reason through complex decisions and document the process.

### 6. Search Before Implementing
Use Brave Search and GitHub to find existing solutions before implementing from scratch.

### 7. Fetch Latest Documentation
Regularly fetch updated documentation from framework websites to stay current.

### 8. Scrape Responsibly
When using Puppeteer, respect website terms of service and rate limits.

### 9. Backup Important Data
Use Filesystem and Git to create regular backups of important models and data.

### 10. Monitor Production
Use PostgreSQL to continuously monitor production metrics and model performance.

---

## Advanced Usage

### Custom Server Configuration
You can modify server configurations in [`.kilocode/mcp.json`](../.kilocode/mcp.json) to add custom arguments or environment variables.

### Server Chaining
Combine multiple servers in workflows:
- Search → Fetch → Analyze → Store
- Scrape → Process → Train → Evaluate → Deploy

### Memory Graphs
Use the Memory server to create knowledge graphs linking experiments, models, and results.

### Sequential Analysis
Use Sequential Thinking for complex multi-step analysis and decision-making.

---

## Security Considerations

1. **API Keys**: Never commit API keys to Git. Use environment variables.
2. **Database Security**: Use strong passwords for PostgreSQL.
3. **Isolated Execution**: Always use E2B for untrusted code.
4. **Access Control**: Limit filesystem access to necessary directories only.
5. **Rate Limiting**: Respect API rate limits when using Fetch and Puppeteer.

---

## Performance Optimization

1. **Cache Documentation**: Cache frequently accessed documentation locally.
2. **Batch Operations**: Batch database queries for better performance.
3. **Parallel Execution**: Run independent experiments in parallel.
4. **Memory Management**: Regularly clean up unused memory entities.
5. **Index Database**: Create indexes on frequently queried PostgreSQL tables.

---

## Maintenance

### Regular Tasks
- Update MCP servers regularly: `npm update -g @modelcontextprotocol/server-*`
- Clean up old experiments from Memory
- Archive old model files
- Backup PostgreSQL database regularly

### Monitoring
- Monitor disk space for model files
- Track PostgreSQL database size
- Monitor E2B API usage
- Check Git repository size

---

## Support and Resources

### Documentation
- [MCP Official Documentation](https://modelcontextprotocol.io)
- [ML Engine Project README](../README.md)
- [Project Architecture](PROJECT_ARCHITECTURE.md)

### Community
- GitHub Issues for bug reports
- Stack Overflow for troubleshooting
- ML Engine Discord/Slack for community support

### Framework Documentation
- [TensorFlow](https://www.tensorflow.org)
- [PyTorch](https://pytorch.org)
- [XGBoost](https://xgboost.readthedocs.io)
- [LightGBM](https://lightgbm.readthedocs.io)
- [scikit-learn](https://scikit-learn.org)

---

## Conclusion

This MCP configuration provides a comprehensive control center for all AI and ML tasks in the ML Engine project. By integrating 10 specialized servers, it covers the entire machine learning lifecycle from data preprocessing to deployment, enabling efficient experimentation, reproducibility, and production deployment of trading models.

For questions or issues, refer to the troubleshooting section or consult the official MCP documentation.
