    def cleanup(self):
        """Clean up resources used during model merging."""
        try:
            # Clear population
            self.population.clear()
            
            # Clear history
            self.history.clear()
            
            # Clear GPU memory if used
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Successfully cleaned up model merging resources")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()