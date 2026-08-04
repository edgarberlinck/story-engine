"""
Network speed measurement utilities for the story-engine project.
"""

import speedtest

def get_network_speed():
    """Get current network download and upload speeds in Mbps.
    
    Returns:
        tuple: (download_mbps, upload_mbps) or (None, None) if failed
    """
    try:
        st = speedtest.Speedtest()
        st.get_best_server()
        
        download = st.download() / 1_000_000
        upload = st.upload() / 1_000_000
        
        return (download, upload)
    except Exception as e:
        print(f"Failed to measure network speed: {e}")
        return (None, None)

def get_max_parallel_downloads():
    """Determine maximum number of parallel downloads based on network speed.
    
    Returns:
        int: Recommended number of parallel downloads (1-8)
    """
    download_speed, _ = get_network_speed()
    
    if download_speed is None:
        return 4  # Default fallback
    
    # Adjust max parallel downloads based on download speed
    if download_speed > 50:  # Fast connection
        return 8
    elif download_speed > 20:  # Medium connection
        return 6
    elif download_speed > 5:   # Slow connection
        return 4
    else:  # Very slow connection
        return 2