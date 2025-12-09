"""
Unit tests for improved utility functions
"""

import unittest
import tempfile
import yaml
from pathlib import Path
import sys
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_config, validate_config, get_config, merge_config


class TestLoadConfig(unittest.TestCase):
    """Test load_config function improvements"""
    
    def setUp(self):
        """Set up temporary config file"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        
        self.valid_config = {
            'model': {
                'hidden_size': 128,
                'num_layers': 3,
                'dropout': 0.2
            },
            'training': {
                'epochs': 100,
                'batch_size': 32
            },
            'data': {
                'sequence_length': 60
            }
        }
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_valid_config(self):
        """Test loading valid configuration"""
        # Write valid config
        with open(self.config_path, 'w') as f:
            yaml.dump(self.valid_config, f)
        
        config = load_config(str(self.config_path))
        
        self.assertIsInstance(config, dict)
        self.assertIn('model', config)
        self.assertIn('training', config)
    
    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            load_config(str(Path(self.temp_dir) / "nonexistent.yaml"))
    
    def test_load_empty_file_raises_error(self):
        """Test that loading empty file raises ValueError"""
        # Create empty file
        empty_path = Path(self.temp_dir) / "empty.yaml"
        empty_path.touch()
        
        with self.assertRaises(ValueError):
            load_config(str(empty_path))
    
    def test_load_invalid_yaml_raises_error(self):
        """Test that invalid YAML raises YAMLError"""
        invalid_path = Path(self.temp_dir) / "invalid.yaml"
        
        with open(invalid_path, 'w') as f:
            f.write("invalid: yaml: content: ][")
        
        with self.assertRaises(yaml.YAMLError):
            load_config(str(invalid_path))
    
    def test_load_non_dict_config_raises_error(self):
        """Test that non-dict config raises ValueError"""
        list_config_path = Path(self.temp_dir) / "list_config.yaml"
        
        with open(list_config_path, 'w') as f:
            yaml.dump([1, 2, 3], f)
        
        with self.assertRaises(ValueError):
            load_config(str(list_config_path))


class TestValidateConfig(unittest.TestCase):
    """Test validate_config function"""
    
    def test_valid_config_passes(self):
        """Test that valid config passes validation"""
        config = {
            'model': {
                'hidden_size': 128,
                'num_layers': 3
            },
            'training': {
                'epochs': 100
            },
            'data': {
                'sequence_length': 60
            }
        }
        
        result = validate_config(config)
        self.assertTrue(result)
    
    def test_missing_section_fails(self):
        """Test that missing section fails validation"""
        config = {
            'model': {
                'hidden_size': 128,
                'num_layers': 3
            }
            # Missing 'training' and 'data' sections
        }
        
        result = validate_config(config)
        self.assertFalse(result)
    
    def test_missing_field_fails(self):
        """Test that missing required field fails validation"""
        config = {
            'model': {
                'hidden_size': 128
                # Missing 'num_layers'
            },
            'training': {
                'epochs': 100
            },
            'data': {
                'sequence_length': 60
            }
        }
        
        result = validate_config(config)
        self.assertFalse(result)
    
    def test_invalid_epochs_fails(self):
        """Test that invalid epochs value fails validation"""
        config = {
            'model': {
                'hidden_size': 128,
                'num_layers': 3
            },
            'training': {
                'epochs': 0  # Invalid
            },
            'data': {
                'sequence_length': 60
            }
        }
        
        result = validate_config(config)
        self.assertFalse(result)
    
    def test_invalid_hidden_size_fails(self):
        """Test that invalid hidden_size fails validation"""
        config = {
            'model': {
                'hidden_size': -10,  # Invalid
                'num_layers': 3
            },
            'training': {
                'epochs': 100
            },
            'data': {
                'sequence_length': 60
            }
        }
        
        result = validate_config(config)
        self.assertFalse(result)
    
    def test_invalid_num_layers_fails(self):
        """Test that invalid num_layers fails validation"""
        config = {
            'model': {
                'hidden_size': 128,
                'num_layers': 0  # Invalid
            },
            'training': {
                'epochs': 100
            },
            'data': {
                'sequence_length': 60
            }
        }
        
        result = validate_config(config)
        self.assertFalse(result)


class TestMergeConfig(unittest.TestCase):
    """Test merge_config function"""
    
    def test_simple_merge(self):
        """Test simple config merge"""
        default = {
            'a': 1,
            'b': 2
        }
        override = {
            'b': 3,
            'c': 4
        }
        
        merged = merge_config(override, default)
        
        self.assertEqual(merged['a'], 1)
        self.assertEqual(merged['b'], 3)  # Override value
        self.assertEqual(merged['c'], 4)
    
    def test_nested_merge(self):
        """Test nested config merge"""
        default = {
            'model': {
                'hidden_size': 128,
                'num_layers': 3,
                'dropout': 0.2
            }
        }
        override = {
            'model': {
                'hidden_size': 256  # Override only this
            }
        }
        
        merged = merge_config(override, default)
        
        self.assertEqual(merged['model']['hidden_size'], 256)
        self.assertEqual(merged['model']['num_layers'], 3)
        self.assertEqual(merged['model']['dropout'], 0.2)
    
    def test_deep_nested_merge(self):
        """Test deeply nested config merge"""
        default = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': 1
                    }
                }
            }
        }
        override = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': 2
                    }
                }
            }
        }
        
        merged = merge_config(override, default)
        
        self.assertEqual(merged['level1']['level2']['level3']['value'], 2)


class TestGetConfig(unittest.TestCase):
    """Test get_config function"""
    
    def setUp(self):
        """Set up temporary config file"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        
        self.user_config = {
            'model': {
                'hidden_size': 256
            },
            'training': {
                'epochs': 50
            },
            'data': {
                'sequence_length': 60
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.user_config, f)
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_config_without_default(self):
        """Test getting config without default"""
        config = get_config(str(self.config_path), validate=False)
        
        self.assertEqual(config['model']['hidden_size'], 256)
        self.assertEqual(config['training']['epochs'], 50)
    
    def test_get_config_with_default(self):
        """Test getting config with default values"""
        default = {
            'model': {
                'hidden_size': 128,
                'num_layers': 3
            },
            'training': {
                'epochs': 100,
                'batch_size': 32
            },
            'data': {
                'sequence_length': 60
            }
        }
        
        config = get_config(str(self.config_path), default_config=default, validate=False)
        
        # User config values should override
        self.assertEqual(config['model']['hidden_size'], 256)
        self.assertEqual(config['training']['epochs'], 50)
        
        # Default values should be preserved
        self.assertEqual(config['model']['num_layers'], 3)
        self.assertEqual(config['training']['batch_size'], 32)
    
    def test_get_config_with_validation(self):
        """Test getting config with validation enabled"""
        # This should work without raising errors
        config = get_config(str(self.config_path), validate=True)
        
        self.assertIsInstance(config, dict)


if __name__ == '__main__':
    unittest.main()
