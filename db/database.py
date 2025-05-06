class Database:
    """
    Main database interface that simulates a real database
    but actually just serves static data with the ability to update it in memory.
    """
    def __init__(self):
        self._data = {}
        self._registries = {}
    
    def register(self, namespace, registry):
        """Register a data registry for a specific namespace"""
        self._registries[namespace] = registry
        if namespace not in self._data:
            self._data[namespace] = {}
    
    def get(self, namespace, key=None, filter_func=None):
        """
        Get data from the database
        
        Args:
            namespace: The data namespace
            key: Optional key to get specific data
            filter_func: Optional filter function
            
        Returns:
            The requested data
        """
        if namespace not in self._data:
            return None
            
        if key is not None:
            return self._data[namespace].get(key)
            
        if filter_func is not None:
            # This simulates a database query with a filter
            if isinstance(self._data[namespace], dict):
                return {k: v for k, v in self._data[namespace].items() 
                        if filter_func(v)}
            elif isinstance(self._data[namespace], list):
                return [item for item in self._data[namespace] 
                        if filter_func(item)]
                        
        return self._data[namespace]
    
    def set(self, namespace, key, value):
        """
        Set data in the database
        
        Args:
            namespace: The data namespace
            key: The data key
            value: The data value to set
        """
        if namespace not in self._data:
            self._data[namespace] = {}
            
        self._data[namespace][key] = value
        return value
    
    def update(self, namespace, key, value):
        """
        Update existing data in the database
        
        Args:
            namespace: The data namespace
            key: The data key
            value: The value to update with
        """
        if namespace not in self._data or key not in self._data[namespace]:
            return None
            
        if isinstance(self._data[namespace][key], dict) and isinstance(value, dict):
            self._data[namespace][key].update(value)
            return self._data[namespace][key]
        else:
            return self.set(namespace, key, value)
    
    def delete(self, namespace, key):
        """
        Delete data from the database
        
        Args:
            namespace: The data namespace
            key: The data key to delete
        """
        if namespace in self._data and key in self._data[namespace]:
            deleted = self._data[namespace][key]
            del self._data[namespace][key]
            return deleted
        return None
    
    def add_to_list(self, namespace, item, id_key='id'):
        """
        Add an item to a list namespace
        
        Args:
            namespace: The data namespace (must be a list)
            item: The item to add
            id_key: The key to use as ID
            
        Returns:
            The added item
        """
        if namespace not in self._data:
            self._data[namespace] = []
            
        if not isinstance(self._data[namespace], list):
            self._data[namespace] = []
            
        self._data[namespace].append(item)
        return item
    
    def initialize(self, namespace, init_func):
        """
        Initialize a namespace with data if it doesn't exist
        
        Args:
            namespace: The data namespace
            init_func: Function that returns initial data
        """
        if namespace not in self._data or not self._data[namespace]:
            self._data[namespace] = init_func()
        return self._data[namespace]