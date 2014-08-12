CREATE DATABASE IF NOT EXISTS 311db_till2013;
USE 311db_till2013;

DROP TABLE IF EXISTS complaints;

-- Data was preprocessed to remove empty cells using perl terminal commant
--  ('' ---> '\N', 'Unspecified' ---> '\N', 'N\/A' ---> '\N')

CREATE TABLE complaints (
    unique_key INT,
    created_date VARCHAR(22),
    closed_date VARCHAR(22),
    agency ENUM ('HPD','DOT','NYPD','DEP','DSNY','DOB','DPR','DOHMH','DCA','TLC','DOF','3-1-1','FDNY','HRA','DOE','DOITT','EDC','DFTA','OPS','CHALL','DHS','OEM','DOP','OATH','COIB','MOIA','FUND','CAU','MOC','OCHIA','DCAS','DYCD','CCRB','MOFTB','OAE','VAC','CWI','EMTF','ART','WF1','SBS','MOVA','DCLA','NYCERS','LAW','UNCC','OCME','ACS','DV','DCP','NYCPPF','DORIS','AJC','NYCHA','LOFT','NYCOOA','OMB','OPA','DOC'),
    -- agency VARCHAR(6),
    agency_name VARCHAR(91),
    complaint_type VARCHAR(41),
    descriptor VARCHAR(106),
    location_type VARCHAR(36),
    -- incident_zip DECIMAL(5),
    incident_zip VARCHAR(10),
    incident_address VARCHAR(81),
    street_name VARCHAR(80),
    cross_street_1 VARCHAR(36),
    cross_street_2 VARCHAR(36),
    intersection_street_1 VARCHAR(38),
    intersection_street_2 VARCHAR(47),
    address_type VARCHAR(12),
    city VARCHAR(32),
    landmark VARCHAR(48),
    facility_type VARCHAR(15),
    status ENUM ('Closed','Open','Pending','Assigned','Email Sent','Started','Closed - No Response Needed','Closed - Email Sent','Unassigned','To Be Rerouted','Closed - Testing','In Progress','Closed - Insufficient Info','Closed - By Phone','Closed - Other','Cancelled','Unable To Respond','Draft','Closed - Letter Sent','Closed - In-Person','In Progress - Needs Approval'),
    -- status VARCHAR(28),
    due_date VARCHAR(22),
    resolution_action_updated_date VARCHAR(22),
    community_board VARCHAR(25),
    borough ENUM ('BROOKLYN','QUEENS','MANHATTAN','BRONX','STATEN ISLAND'),
    x_coordinate_state_plane VARCHAR(7),
    y_coordinate_state_plane VARCHAR(7),
    park_facility_name VARCHAR(95),
    park_borough ENUM ('BROOKLYN','QUEENS','MANHATTAN','BRONX','STATEN ISLAND'),
    school_name VARCHAR(95),
    school_number VARCHAR(8),
    school_region ENUM('Region 6','Region 9','Region 4','Region 1','Region 3','Region 10','Region 7','Region 8','Region 2','Region 5','Alternative Superintendency'),
    -- school_region VARCHAR(27),
    school_code VARCHAR(6),
    school_phone_number VARCHAR(10),
    school_address VARCHAR(120),
    school_city VARCHAR(19),
    school_state ENUM('NY'),
    school_zip DECIMAL,
    school_not_found ENUM('','Y','N'),
    school_or_citywide_complaint VARCHAR(18),
    vehicle_type ENUM('','Car Service','Ambulette / Paratransit','Commuter Van'),
    taxi_company_borough ENUM ('BROOKLYN','QUEENS','MANHATTAN','BRONX','NULL','STATEN ISLAND'),
    taxi_pick_up_location ENUM('Other','','JFK Airport','La Guardia Airport','Grand Central Station','New York-Penn Station','Port Authority Bus Terminal','Intersection'),
    bridge_highway_name VARCHAR(42),
    bridge_highway_direction VARCHAR(30),
    road_ramp ENUM('','Roadway','Ramp','N/A'),
    bridge_highway_segment VARCHAR(100),
    garage_lot_name VARCHAR(27),
    ferry_direction VARCHAR(19),
    ferry_terminal_name VARCHAR(95),
    latitude VARCHAR(18),
    longitude VARCHAR(18),
    location VARCHAR(40)
);

-- LOAD DATA LOCAL INFILE '/Users/dahlia/Documents/Insight/Project311/311Data/311_Service_Requests_from_2010_to_Present_2.csv'
LOAD DATA LOCAL INFILE '/Users/dahlia/Documents/Insight/Project311/311Data/311_Service_Requests_from_2010_to_2013.csv'
    -- IGNORE
    INTO TABLE complaints
    FIELDS TERMINATED BY ',' ENCLOSED BY '"'
    LINES TERMINATED BY '\n'
    IGNORE 1 LINES;