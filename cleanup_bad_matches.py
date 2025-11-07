"""
Script to remove matches containing p1/p2/p3/p4 demo files from the database.
"""

from research_demo_scraper.cs2_demo_database import get_session, Tournament, Match, DemoFile

# List of (tournament_id, match_id) pairs with bad demo files
# Note: Both IDs are strings in the database
bad_matches = [
    ("7902", "2382434"), ("7902", "2382595"), ("7902", "2382605"), ("7902", "2382612"), ("7902", "2382613"),
    ("7905", "2381907"),
    ("8036", "2381637"), ("8036", "2381639"), ("8036", "2381640"), ("8036", "2381642"), ("8036", "2381644"),
    ("8036", "2381759"), ("8036", "2381762"), ("8036", "2381767"),
    ("8037", "2382375"), ("8037", "2382376"), ("8037", "2382377"), ("8037", "2382687"), ("8037", "2382690"),
    ("8038", "2383761"), ("8038", "2383763"), ("8038", "2383826"), ("8038", "2383827"), ("8038", "2383833"),
    ("8038", "2383834"), ("8038", "2383835"), ("8038", "2383837"), ("8038", "2383840"), ("8038", "2383841"),
    ("8038", "2383842"), ("8038", "2383844"),
    ("8039", "2384299"), ("8039", "2384301"), ("8039", "2384302"), ("8039", "2384383"), ("8039", "2384387"),
    ("8039", "2384388"),
    ("8040", "2385931"), ("8040", "2385934"), ("8040", "2385935"), ("8040", "2385937"), ("8040", "2385940"),
    ("8040", "2385942"), ("8040", "2385948"), ("8040", "2385953"), ("8040", "2385955"), ("8040", "2385956"),
    ("8040", "2385965"), ("8040", "2385969"), ("8040", "2385970"),
    ("8045", "2382221"), ("8045", "2382229"), ("8045", "2382236"), ("8045", "2382240"), ("8045", "2382241"),
    ("8046", "2387320"), ("8046", "2387333"), ("8046", "2387334"),
    ("8063", "2383593"), ("8063", "2383594"),
    ("8064", "2385296"), ("8064", "2385305"), ("8064", "2385307"), ("8064", "2385318"), ("8064", "2385326"),
    ("8358", "2382115"), ("8358", "2382118"), ("8358", "2382119"),
    ("8539", "2385394"), ("8539", "2385395"), ("8539", "2385397"), ("8539", "2385402"), ("8539", "2385403"),
    ("8539", "2385404"), ("8539", "2385406")
]

def cleanup_bad_matches():
    """Remove matches with bad demo files from the database."""
    session = get_session()
    
    try:
        total_demo_files_deleted = 0
        total_matches_deleted = 0
        
        print(f"\nRemoving {len(bad_matches)} matches with bad demo files...")
        print("=" * 60)
        
        for tournament_id, match_id in bad_matches:
            # Find the match
            match = session.query(Match).filter(
                Match.tournament_id == tournament_id,
                Match.match_id == match_id
            ).first()
            
            if match:
                # Count and delete associated demo files
                demo_files = session.query(DemoFile).filter(
                    DemoFile.match_id == match.match_id
                ).all()
                
                demo_count = len(demo_files)
                for demo in demo_files:
                    session.delete(demo)
                
                # Delete the match
                session.delete(match)
                
                total_demo_files_deleted += demo_count
                total_matches_deleted += 1
                
                print(f"Tournament {tournament_id}, Match {match_id}: "
                      f"Deleted {demo_count} demo file(s)")
            else:
                print(f"Tournament {tournament_id}, Match {match_id}: "
                      f"Not found in database (may have been already deleted)")
        
        # Commit all deletions
        session.commit()
        
        print("=" * 60)
        print(f"\nSummary:")
        print(f"  Matches deleted: {total_matches_deleted}")
        print(f"  Demo file records deleted: {total_demo_files_deleted}")
        print(f"\nDatabase cleanup complete!")
        
        # Show remaining counts
        remaining_matches = session.query(Match).count()
        remaining_demos = session.query(DemoFile).count()
        remaining_tournaments = session.query(Tournament).count()
        
        print(f"\nRemaining in database:")
        print(f"  Tournaments: {remaining_tournaments}")
        print(f"  Matches: {remaining_matches}")
        print(f"  Demo files: {remaining_demos}")
        
    except Exception as e:
        session.rollback()
        print(f"\nError during cleanup: {e}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    cleanup_bad_matches()
