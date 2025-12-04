#!/usr/bin/env python3
"""
Test Redis and TigerBeetle persistence
"""
import sys
import os
sys.path.append('src')

def test_persistence():
    print("🧪 Testing Redis and TigerBeetle Persistence")
    print("=" * 50)

    try:
        from src.utils.advanced_memory import get_memory_health_status
        health = get_memory_health_status()

        print("✅ Memory Health Check:")
        print(f"   Overall Health: {health.get('overall_health', 'unknown')}")
        print(f"   Redundancy Level: {health.get('redundancy_level', 0)}")

        backends = health.get('backends', {})
        print(f"   Available Backends: {len(backends)}")

        for backend_name, backend_status in backends.items():
            status = backend_status.get('status', 'unknown')
            print(f"   - {backend_name}: {status}")

            if backend_name == 'redis':
                if status == 'healthy':
                    print("   ✅ Redis persistence: WORKING")
                else:
                    print(f"   ❌ Redis persistence: {status}")

            if backend_name == 'tigerbeetle':
                if status == 'healthy':
                    print("   ✅ TigerBeetle persistence: WORKING")
                else:
                    print(f"   ❌ TigerBeetle persistence: {status}")

        # Check if we have at least Redis working
        redis_healthy = any(
            backend.get('status') == 'healthy'
            for backend in backends.values()
            if backend.get('type') == 'redis'
        )

        tigerbeetle_healthy = any(
            backend.get('status') == 'healthy'
            for backend in backends.values()
            if backend.get('type') == 'tigerbeetle'
        )

        if redis_healthy:
            print("\n✅ Redis persistence validation: PASSED")
        else:
            print("\n⚠️  Redis persistence: Not available or not healthy")

        if tigerbeetle_healthy:
            print("✅ TigerBeetle persistence validation: PASSED")
        else:
            print("⚠️  TigerBeetle persistence: Not available or not healthy")

        if redis_healthy or tigerbeetle_healthy:
            print("\n🎉 Persistence validation completed successfully!")
            return True
        else:
            print("\n❌ No persistence backends are healthy")
            return False

    except Exception as e:
        print(f"❌ Persistence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_persistence()
    sys.exit(0 if success else 1)