"""Test database connection and verify we can write data"""
from research_demo_scraper.cs2_demo_database import get_session, Tournament, Match, DemoFile
from datetime import datetime

print("Testing database connection...")
session = get_session()

# Try to create a test tournament
print("\n1. Creating test tournament...")
test_tournament = Tournament(
    tournament_id="TEST_001",
    tournament_name="Test Tournament"
)
session.add(test_tournament)
session.commit()
print("✓ Tournament created")

# Try to create a test match
print("\n2. Creating test match...")
test_match = Match(
    match_id="TEST_MATCH_001",
    tournament_id="TEST_001",
    team1_name="Test Team 1",
    team2_name="Test Team 2",
    team1_score=2,
    team2_score=1,
    best_of=3,
    event_stage="Test Stage",
    match_url="https://test.com"
)
session.add(test_match)
session.commit()
print("✓ Match created")

# Try to create a test demo file
print("\n3. Creating test demo file...")
test_demo = DemoFile(
    match_id="TEST_MATCH_001",
    tournament_id="TEST_001",
    map_name="test_map",
    demo_filename="test.dem",
    demo_path="/path/to/test.dem",
    demo_size_mb=100.5,
    is_downloaded=True,
    is_parsed=False
)
session.add(test_demo)
session.commit()
print("✓ Demo file created")

# Query to verify
print("\n4. Verifying data...")
tournament_count = session.query(Tournament).count()
match_count = session.query(Match).count()
demo_count = session.query(DemoFile).count()

print(f"   Tournaments: {tournament_count}")
print(f"   Matches: {match_count}")
print(f"   Demo Files: {demo_count}")

# Clean up test data
print("\n5. Cleaning up test data...")
session.query(DemoFile).filter_by(tournament_id="TEST_001").delete()
session.query(Match).filter_by(tournament_id="TEST_001").delete()
session.query(Tournament).filter_by(tournament_id="TEST_001").delete()
session.commit()
print("✓ Test data removed")

print("\n✅ Database connection test PASSED - everything working!")
session.close()
