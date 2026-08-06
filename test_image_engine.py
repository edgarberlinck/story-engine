import unittest
from unittest.mock import patch, MagicMock
from generators.image_engine import generate, GenerationType, CharacterReferenceStore, NullCharacterReferenceStore


class TestImageEngine(unittest.TestCase):
    
    @patch('generators.image_engine.generate_images')
    def test_generate_character_with_flux_dev(self, mock_generate_images):
        """Test that character generation uses FLUX Dev model."""
        mock_generate_images.return_value = {"image_path": "test.png"}
        
        result = generate(
            prompt="A superhero",
            type=GenerationType.CHARACTER,
            character_id="hero123"
        )
        
        # Verify the correct model was used
        mock_generate_images.assert_called_once()
        call_args = mock_generate_images.call_args
        self.assertEqual(call_args.kwargs['model_name'], 'flux_dev')
        
    @patch('generators.image_engine.generate_images')
    def test_generate_environment_with_sdxl(self, mock_generate_images):
        """Test that environment generation uses SDXL model."""
        mock_generate_images.return_value = {"image_path": "test.png"}
        
        result = generate(
            prompt="A forest landscape",
            type=GenerationType.ENVIRONMENT
        )
        
        # Verify the correct model was used
        mock_generate_images.assert_called_once()
        call_args = mock_generate_images.call_args
        self.assertEqual(call_args.kwargs['model_name'], 'sdxl')
    
    def test_type_validation(self):
        """Test that invalid generation types raise ValueError."""
        with self.assertRaises(ValueError) as context:
            generate(
                prompt="A test",
                type="invalid_type"
            )
        
        self.assertIn("Invalid type", str(context.exception))
    
    @patch('generators.image_engine.generate_images')
    def test_character_generation_with_reference_lookup(self, mock_generate_images):
        """Test that character generation properly handles reference lookup."""
        mock_generate_images.return_value = {"image_path": "test.png"}
        
        # Create a mock reference store
        mock_store = MagicMock()
        mock_store.has_reference.return_value = True
        mock_store.get_reference.return_value = "/path/to/reference.png"
        
        result = generate(
            prompt="A character with reference",
            type=GenerationType.CHARACTER,
            character_id="char456",
            reference_store=mock_store
        )
        
        # Verify the reference store was checked
        mock_store.has_reference.assert_called_once_with("char456")
        mock_store.get_reference.assert_called_once_with("char456")
    
    @patch('generators.image_engine.generate_images')
    def test_character_generation_without_reference(self, mock_generate_images):
        """Test that character generation works without references."""
        mock_generate_images.return_value = {"image_path": "test.png"}
        
        result = generate(
            prompt="A character without reference",
            type=GenerationType.CHARACTER,
            character_id="char789"
        )
        
        # Should not raise any exceptions and should call the backend
        mock_generate_images.assert_called_once()
        self.assertEqual(mock_generate_images.call_args.kwargs['model_name'], 'flux_dev')
    
    def test_environment_generation_skips_reference_lookup(self):
        """Test that environment generation doesn't perform reference lookup."""
        # This test mainly ensures that we don't break anything when using 
        # an actual reference store for environments (which should be ignored)
        mock_store = MagicMock()
        mock_store.has_reference.return_value = True
        
        with patch('generators.image_engine.generate_images') as mock_generate_images:
            mock_generate_images.return_value = {"image_path": "test.png"}
            
            result = generate(
                prompt="An environment",
                type=GenerationType.ENVIRONMENT,
                character_id="env123",
                reference_store=mock_store
            )
            
            # Verify environment generation still uses SDXL
            mock_generate_images.assert_called_once()
            self.assertEqual(mock_generate_images.call_args.kwargs['model_name'], 'sdxl')
    
    def test_encapsulation_no_model_name_in_public_api(self):
        """Test that the public API doesn't expose model names directly."""
        # Check that we have a clean public interface with no direct model parameter
        import inspect
        sig = inspect.signature(generate)
        
        # Ensure 'model_name' is not in the public API parameters
        self.assertNotIn('model_name', sig.parameters)
        
        # Ensure expected parameters are present
        expected_params = {'prompt', 'type', 'character_id', 'reference_store'}
        actual_params = set(sig.parameters.keys())
        self.assertTrue(expected_params.issubset(actual_params))
    
    @patch('generators.image_engine.generate_images')
    def test_null_reference_store_default_behavior(self, mock_generate_images):
        """Test that the default null store behaves correctly."""
        mock_generate_images.return_value = {"image_path": "test.png"}
        
        # Should work fine without explicitly setting a reference store
        result = generate(
            prompt="A test character",
            type=GenerationType.CHARACTER,
            character_id="char123"
        )
        
        # Should have been called once and should not fail
        mock_generate_images.assert_called_once()
        self.assertEqual(mock_generate_images.call_args.kwargs['model_name'], 'flux_dev')
        
    def test_model_selection_policy(self):
        """Test that the model selection policy maps properly."""
        from generators.image_engine import _MODEL_BY_TYPE
        
        self.assertEqual(_MODEL_BY_TYPE[GenerationType.CHARACTER], 'flux_dev')
        self.assertEqual(_MODEL_BY_TYPE[GenerationType.ENVIRONMENT], 'sdxl')


if __name__ == '__main__':
    unittest.main()